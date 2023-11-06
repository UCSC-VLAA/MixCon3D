import torch
import numpy as np
import wandb
import logging
import os
import torch.distributed.nn
import torch.nn.functional as F
from tqdm import tqdm
import math
from utils.misc import get_autocast, unwrap_model

class Trainer(object):
    def __init__(self, rank, config, model, text_logit_scale, image_logit_scale, img_text_logit_scale, pc_img_to_text_logit_scale, image_proj,
                 text_proj, pc_img_to_text_proj, multi_view_proj, optimizer, scheduler, train_loader, modelnet40_loader, objaverse_lvis_loader=None,
                 scanobjectnn_loader=None, model_ema=None, image_proj_ema=None, text_proj_ema=None, pc_img_to_text_proj_ema=None, multi_view_proj_ema=None):
        self.rank = rank
        self.config = config

        self.image_logit_scale = image_logit_scale
        self.text_logit_scale = text_logit_scale
        self.img_text_logit_scale = img_text_logit_scale
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.modelnet40_loader = modelnet40_loader
        self.objaverse_lvis_loader = objaverse_lvis_loader
        self.scanobjectnn_loader = scanobjectnn_loader
        self.epoch = 0
        self.step = 0
        self.best_img_contras_acc = 0
        self.best_text_contras_acc = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_lvis_acc = 0
        self.num_batches_per_epoch = len(train_loader)

        # For model and EMA model creating
        self.model = model
        self.image_proj = image_proj
        self.text_proj = text_proj
        self.model_ema = model_ema
        self.image_proj_ema = image_proj_ema
        self.text_proj_ema = text_proj_ema

        # For 2 modals alignment:
        self.pc_img_to_text_proj = pc_img_to_text_proj
        self.pc_img_to_text_logit_scale = pc_img_to_text_logit_scale
        self.pc_img_to_text_proj_ema = pc_img_to_text_proj_ema

        # for multi-view projection
        self.multi_view_proj = multi_view_proj
        self.multi_view_proj_ema = multi_view_proj_ema

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path)

        # For Adam update model
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.config.training.use_text_proj:
            self.text_proj.load_state_dict(checkpoint['text_proj'])
        if self.config.training.use_image_proj:
            self.image_proj.load_state_dict(checkpoint['image_proj'])

        # For EMA update model

        # For statistics loading
        self.image_logit_scale.load_state_dict(checkpoint['image_logit_scale'])
        self.text_logit_scale.load_state_dict(checkpoint['text_logit_scale'])
        if self.config.training.image_text_align:
            self.img_text_logit_scale.load_state_dict(checkpoint['img_text_logit_scale'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.use_openclip_optimizer_scheduler == False:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_img_contras_acc = checkpoint['best_img_contras_acc']
        self.best_text_contras_acc = checkpoint['best_text_contras_acc']
        self.best_modelnet40_overall_acc = checkpoint['best_modelnet40_overall_acc']
        self.best_modelnet40_class_acc = checkpoint['best_modelnet40_class_acc']
        self.best_lvis_acc = checkpoint['best_lvis_acc']

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))
        logging.info("----Best img contras acc: {}".format(self.best_img_contras_acc))
        logging.info("----Best text contras acc: {}".format(self.best_text_contras_acc))
        logging.info("----Best modelnet40 overall acc: {}".format(self.best_modelnet40_overall_acc))
        logging.info("----Best modelnet40 class acc: {}".format(self.best_modelnet40_class_acc))
        logging.info("----Best lvis acc: {}".format(self.best_lvis_acc))

    def contras_loss(self, feat1, feat2, logit_scale=1, mask=None):
        if self.config.ngpu > 1:
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
            all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
            all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
            logits = logit_scale * all_feat1 @ all_feat2.T
        else:
            logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
        if mask is not None:
            logits = logits * mask
        labels = torch.arange(logits.shape[0]).to(self.config.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, accuracy

    def train_one_epoch(self, epoch):
        # Using bfloat16 precision (precision = amp_bfloat16) can reduce a little memory but no much performance gain
        precision = self.config.training.precision
        # (precision = amp_bfloat16)
        autocast = get_autocast(precision)

        self.model.train()
        if self.config.training.use_text_proj:
            self.text_proj.train()
        if self.config.training.use_image_proj:
            self.image_proj.train()

        text_contras_acc_list = []
        img_contras_acc_list = []
        if self.config.training.use_mask:
            k = self.config.dataset.negative_sample_num
            s = self.config.dataset.train_batch_size
            mask1 = np.eye(k * s).astype(np.bool)
            mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
            mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

        # Start to add accumulating batch module

        accum_freq = self.config.dataset.accum_freq
        if accum_freq > 1:
            accum_point, accum_text, accum_image, accum_image_features, \
                accum_text_features, accum_point_features, accum_pc_img_features = [], [],[], [], [], [], []
        num_batches_per_epoch_accum = self.num_batches_per_epoch // accum_freq

        logging.info("----Initial Batch: {0} Accum_Iter: {1} After_Accum_Batch: {2}".format(
            self.num_batches_per_epoch, accum_freq, num_batches_per_epoch_accum))

        for i, data in enumerate(self.train_loader):

            i_accum = i // accum_freq
            self.step = num_batches_per_epoch_accum * epoch + i_accum

            idx = data['has_text_idx']

            if i > 51 and self.config.training.debug:
                break

            img_text_contras_loss, img_text_contras_acc, img_text_logit_scale = torch.tensor(0), \
                torch.tensor(0), torch.tensor(0)

            pc_img_to_text_contras_loss, pc_img_to_text_contras_acc, pc_img_to_text_logit_scale = torch.tensor(0), \
                torch.tensor(0), torch.tensor(0)

            # TODO: Implement the accumulate_iter module here
            mask = None
            if accum_freq == 1:
                with autocast():
                    if not self.config.model.get("use_dense", False):
                        pred_feat = self.model(data['xyz'], data['features'], device=self.config.device,
                                               quantization_size=self.config.model.voxel_size)
                    else:
                        pred_feat = self.model(data['xyz_dense'], data['features_dense'])

                    image_logit_scale, text_logit_scale = self.image_logit_scale(None), self.text_logit_scale(None)

                    text_feat = torch.vstack(data['text_feat']).to(self.config.device)
                    img_feat = torch.vstack(data['img_feat']).to(self.config.device)

                    # Image Feature Pipeline: Frozen CLIP feature --(MLP)--> Individual Feature --(MV)--> MV Features
                    if self.config.training.use_image_proj:
                        if self.config.training.share_proj:
                            img_feat = self.text_proj(img_feat)
                        else:
                            img_feat = self.image_proj(img_feat)

                    if self.config.training.img_loss_mode == "Avg":
                        img_feat = img_feat.mean(dim=1)
                    if self.config.training.img_loss_mode == "Max":
                        img_feat, _ = img_feat.max(dim=1)
                    if self.config.training.img_loss_mode == "avg_proj":
                        img_feat = self.multi_view_proj(img_feat.mean(dim=1))
                    if self.config.training.img_loss_mode == "Max_proj":
                        img_feat, _ = img_feat.max(dim=1)
                        img_feat = self.multi_view_proj(img_feat)

                    # Performing single modality contrastive losses (Point-to-Text and Point-to-Image)
                    if len(idx) > 0:
                        if self.config.training.use_text_proj:
                            text_feat = self.text_proj(text_feat)
                        text_contras_loss, text_contras_acc = self.contras_loss(pred_feat[idx], text_feat,
                                                                                    logit_scale=text_logit_scale, mask=mask)
                    img_contras_loss, img_contras_acc = self.contras_loss(pred_feat, img_feat,
                                                                                  logit_scale=image_logit_scale,
                                                                                  mask=mask)
                    if self.config.training.image_text_align:
                        img_text_logit_scale = self.img_text_logit_scale(None)
                        img_text_contras_loss, img_text_contras_acc = self.contras_loss(img_feat, text_feat,
                                                                                       logit_scale=img_text_logit_scale, mask=mask)
                    if self.config.training.pc_img_to_text:
                        pc_img_feat = self.pc_img_to_text_proj(torch.cat((pred_feat, img_feat), dim=1))
                        pc_img_to_text_logit_scale = self.pc_img_to_text_logit_scale(None)
                        pc_img_to_text_contras_loss, pc_img_to_text_contras_acc = self.contras_loss(pc_img_feat,
                                                                                                            text_feat,
                                                                                                            logit_scale=pc_img_to_text_logit_scale,
                                                                                                            mask=mask)
                    if self.config.training.loss_avg:
                        loss = 1/3*(text_contras_loss + img_contras_loss + img_text_contras_loss) + pc_img_to_text_contras_loss
                    else:
                        loss = text_contras_loss + img_contras_loss + img_text_contras_loss + pc_img_to_text_contras_loss
                loss.backward()
            else:
                # TODO: Implement the accumulated iter here
                with torch.no_grad():
                    # Cache raw data and features
                    # accum_point/text/image are raw input point/text/image data
                    # accum_point/text/image_features are cached features
                    if not self.config.model.get("use_dense", False):
                        pred_feat = self.model(data['xyz'], data['features'], device=self.config.device,
                                               quantization_size=self.config.model.voxel_size)
                        accum_point.append([data['xyz'], data['features']])
                        accum_point_features.append(pred_feat)
                    else:
                        pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                        accum_point.append([data['xyz_dense'], data['features_dense']])
                        accum_point_features.append(pred_feat)

                    text_feat = torch.vstack(data['text_feat']).to(self.config.device)
                    img_feat = torch.vstack(data['img_feat']).to(self.config.device)
                    accum_text.append(text_feat)
                    accum_image.append(img_feat)

                    # Processing the text and image features
                    if self.config.training.use_text_proj:
                        text_feat = self.text_proj(text_feat)

                    if self.config.training.use_image_proj:
                        if self.config.training.share_proj:
                            img_feat = self.text_proj(img_feat)
                        else:
                            img_feat = self.image_proj(img_feat)

                    if self.config.training.img_loss_mode == "Avg":
                        img_feat = img_feat.mean(dim=1)
                    if self.config.training.img_loss_mode == "Max":
                        img_feat, _ = img_feat.max(dim=1)
                    if self.config.training.img_loss_mode == "avg_proj":
                        img_feat = self.multi_view_proj(img_feat.mean(dim=1))
                    if self.config.training.img_loss_mode == "Max_proj":
                        img_feat, _ = img_feat.max(dim=1)
                        img_feat = self.multi_view_proj(img_feat)

                    # Cache the features for single modality inference (used for P <-> I and T <-> I)
                    accum_image_features.append(img_feat)
                    accum_text_features.append(text_feat)

                    if self.config.training.pc_img_to_text:
                        pc_img_feat = self.pc_img_to_text_proj(torch.cat((pred_feat, img_feat), dim=1))
                        accum_pc_img_features.append(pc_img_feat)

                # If (i + 1) % accum_freq is not zero, move on to the next batch.
                if ((i + 1) % accum_freq) > 0:
                    continue
                # Now, we have point (accum_point), accum_point_features,
                # text (accum_text_features), image (accum_image_features)
                # Now, ready to take gradients for the last accum_freq batches.
                # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
                # Call backwards each time, but only step optimizer at the end.
                self.optimizer.zero_grad()
                for j in range(accum_freq):
                    # Forward the point cloud data
                    point_xyz, point_features = accum_point[j]
                    if not self.config.model.get("use_dense", False):
                        pred_feat = self.model(point_xyz, point_features, device=self.config.device,
                                               quantization_size=self.config.model.voxel_size)
                    else:
                        pred_feat = self.model(point_xyz, point_features)

                    # Forward the image and text data
                    text_feat, img_feat = accum_text[j], accum_image[j]
                    if self.config.training.use_text_proj:
                        text_feat = self.text_proj(text_feat)

                    # Processing the image features
                    if self.config.training.use_image_proj:
                        if self.config.training.share_proj:
                            img_feat = self.text_proj(img_feat)
                        else:
                            img_feat = self.image_proj(img_feat)

                    if self.config.training.img_loss_mode == "Avg":
                        img_feat = img_feat.mean(dim=1)
                    if self.config.training.img_loss_mode == "Max":
                        img_feat, _ = img_feat.max(dim=1)
                    if self.config.training.img_loss_mode == "avg_proj":
                        img_feat = self.multi_view_proj(img_feat.mean(dim=1))
                    if self.config.training.img_loss_mode == "Max_proj":
                        img_feat, _ = img_feat.max(dim=1)
                        img_feat = self.multi_view_proj(img_feat)

                    # Replace the features in the feature caches
                    point_feats = torch.cat(accum_point_features[:j] + [pred_feat] + accum_point_features[j + 1:])
                    text_feats = torch.cat(accum_text_features[:j] + [text_feat] + accum_text_features[j + 1:])
                    image_feats = torch.cat(accum_image_features[:j] + [img_feat] + accum_image_features[j + 1:])
                    if self.config.training.pc_img_to_text:
                        pc_img_feat = self.pc_img_to_text_proj(torch.cat((pred_feat, img_feat), dim=1))
                        pc_img_feats = torch.cat(accum_pc_img_features[:j] + [pc_img_feat] + accum_pc_img_features[j + 1:])

                    # For contrastive learning losses:
                    image_logit_scale, text_logit_scale = self.image_logit_scale(None), self.text_logit_scale(None)
                    text_contras_loss, text_contras_acc = self.contras_loss(point_feats, text_feats,
                                                                            logit_scale=text_logit_scale, mask=None)
                    img_contras_loss, img_contras_acc = self.contras_loss(point_feats, image_feats,
                                                                          logit_scale=image_logit_scale, mask=None)
                    if self.config.training.image_text_align:
                        img_text_logit_scale = self.img_text_logit_scale(None)
                        img_text_contras_loss, img_text_contras_acc = self.contras_loss(image_feats, text_feats,
                                                                                       logit_scale=img_text_logit_scale, mask=mask)
                    if self.config.training.pc_img_to_text:
                        pc_img_to_text_logit_scale = self.pc_img_to_text_logit_scale(None)
                        pc_img_to_text_contras_loss, pc_img_to_text_contras_acc = self.contras_loss(pc_img_feats,
                                                                                                            text_feats,
                                                                                                            logit_scale=pc_img_to_text_logit_scale,
                                                                                                            mask=mask)
                    if self.config.training.loss_avg:
                        loss = 1/3*(text_contras_loss + img_contras_loss + img_text_contras_loss) + pc_img_to_text_contras_loss
                    else:
                        loss = text_contras_loss + img_contras_loss + img_text_contras_loss + pc_img_to_text_contras_loss
                    loss.backward()

            # Before accumulating batch
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.model_ema is not None:
                self.model_ema.update(self.model)
                self.text_proj_ema.update(self.text_proj)
                self.image_proj_ema.update(self.image_proj)
                self.multi_view_proj_ema.update(self.multi_view_proj)
                self.pc_img_to_text_proj_ema.update(self.pc_img_to_text_proj)

            if accum_freq > 1:
                accum_point, accum_text, accum_image, accum_image_features, \
                    accum_text_features, accum_point_features, accum_pc_img_features = [], [], [], [], [], [], []

            # Clamp the logit_value to 4.602 = ln(100) by default
            with torch.no_grad():
                unwrap_model(self.image_logit_scale).logit_scale.clamp_(0, math.log(100))
                unwrap_model(self.text_logit_scale).logit_scale.clamp_(0, math.log(100))

            if self.config.training.use_openclip_optimizer_scheduler:
                self.scheduler(self.step)
            else:
                self.scheduler.step()

            text_contras_acc_list.append(text_contras_acc.item())
            img_contras_acc_list.append(img_contras_acc.item())

            # if self.rank == 0 and self.step % max(1, self.config.training.log_freq // accum_freq) == 0:
            if self.rank == 0 and self.step % (self.config.training.log_freq // accum_freq) == 0:
                #try:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/text_contras_loss": text_contras_loss.item() if len(idx) > 0 else 0,
                    "train/img_contras_loss": img_contras_loss.item(),
                    "train/img_text_contras_loss": img_text_contras_loss.item(),
                    "train/pc_img_to_text_contras_loss": pc_img_to_text_contras_loss.item(),
                    "train/text_contras_acc": text_contras_acc.item() if len(idx) > 0 else 0,
                    "train/img_contras_acc": img_contras_acc.item(),
                    "train/img_text_contras_acc": img_text_contras_acc.item(),
                    "train/pc_img_to_text_contras_acc": pc_img_to_text_contras_acc.item(),
                    "train/lr": self.optimizer.param_groups[0]['lr'],
                    "train/epoch": self.epoch,
                    "train/step": self.step,
                    "train/image_logit_scale": image_logit_scale,
                    "train/text_logit_scale": text_logit_scale,
                    "train/img_text_logit_scale": img_text_logit_scale,
                    "train/has_text": len(idx),
                    "train/filtered_pair": (mask == False).sum() if mask is not None else 0
                })
                #except:
                #    print("wandb log error", flush=True)

        if self.rank == 0:
            logging.info('Train: text_cotras_acc: {0} image_contras_acc: {1}' \
                         .format(np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
                                 np.mean(img_contras_acc_list)))

    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
                logging.info("Epoch: {}".format(self.epoch))
            self.train_one_epoch(epoch)
            if self.rank == 0:
                self.save_model('latest')
                self.test_modelnet40(self.model, self.text_proj, "_sgd")
                self.test_objaverse_lvis(self.model, self.text_proj, self.image_proj, self.pc_img_to_text_proj, self.multi_view_proj, "_sgd")
                self.test_scanobjectnn(self.model, self.text_proj, "_sgd")
                if self.model_ema is not None:
                    self.test_modelnet40(self.model_ema.module, self.text_proj_ema.module, "_ema")
                    self.test_objaverse_lvis(self.model_ema.module, self.text_proj_ema.module, self.image_proj_ema.module, self.pc_img_to_text_proj_ema.module,
                                             self.multi_view_proj, "_ema")
                    self.test_scanobjectnn(self.model_ema.module, self.text_proj_ema.module, "_ema")

            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model('epoch_{}'.format(self.epoch))

    def test_objaverse_lvis(self, model, text_proj, img_proj, pc_img_to_text_proj, multi_view_proj, model_suffix=""):
        model.eval()
        clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            text_proj.eval()
            clip_text_feat = text_proj(clip_text_feat)

        # category2idx = self.objaverse_lvis_loader.dataset.category2idx
        # idx2category = {v: k for k, v in category2idx.items()}
        per_cat_correct, per_cat_count = torch.zeros(1156).to(self.config.device), torch.zeros(1156).to(self.config.device)
        logits_all, logits_img_all, logits_img_mv_all, logits_merge_all, logits_pc_img_all, \
            logits_pc_img_mv_all, labels_all = [], [], [], [], [], [], []

        with torch.no_grad():
            for data in tqdm(self.objaverse_lvis_loader):
                if not self.config.model.get("use_dense", False):
                    pred_feat = model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = model(data['xyz_dense'].cuda(), data['features_dense'].cuda())

                img_feat = torch.vstack(data['img_feat']).to(self.config.device)
                if self.config.training.use_image_proj:
                    img_proj.eval()
                    if self.config.training.share_proj:
                        img_feat = text_proj(img_feat)
                    else:
                        img_feat = img_proj(img_feat)

                # For single view inference
                img_feat_0 = img_feat[:, 0, :]
                img_feat_mv = img_feat[:, 0, :]
                # For multi-view inference:
                if self.config.training.img_loss_mode == "Avg":
                    img_feat_mv = img_feat.mean(dim=1)
                if self.config.training.img_loss_mode == "Max":
                    img_feat_mv, _ = img_feat.max(dim=1)
                if self.config.training.img_loss_mode == "Avg_proj":
                    img_feat_mv = multi_view_proj(img_feat.mean(dim=1))
                if self.config.training.img_loss_mode == "Max_proj":
                    img_feat_mv, _ = img_feat.max(dim=1)
                    img_feat_mv = multi_view_proj(img_feat_mv)

                # For 2 modals inference:
                if self.config.training.pc_img_to_text:
                    pc_img_feat_0 = pc_img_to_text_proj(torch.cat((pred_feat, img_feat_0), dim=1))
                    pc_img_feat_mv = pc_img_to_text_proj(torch.cat((pred_feat, img_feat_mv), dim=1))

                    # for single view pc-img inference
                    logits_pc_img = F.normalize(pc_img_feat_0, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                    logits_pc_img_all.append(logits_pc_img.detach())

                    # for multi view pc-img inference
                    logits_pc_img_mv = F.normalize(pc_img_feat_mv, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                    logits_pc_img_mv_all.append(logits_pc_img_mv.detach())

                # Point-to-Text Logits
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                logits_all.append(logits.detach())

                # Image-to-Text Logits
                logits_img = F.normalize(img_feat_0, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                logits_img_all.append(logits_img.detach())
                logits_img_mv = F.normalize(img_feat_mv, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                logits_img_mv_all.append(logits_img_mv.detach())

                # Avg the logits of Point-to-Text and Image-to-Text.
                logits_merge = logits + logits_img_mv
                logits_merge_all.append(logits_merge)

                # TODO: Point-to-Text + Image-to-Text + (Point-Image)-to-Text logits inference

                # Get the ground truth class labels
                labels = data['category'].to(self.config.device)
                labels_all.append(labels)

        logits_list = [logits_all, logits_img_all, logits_merge_all, logits_pc_img_all, logits_img_mv_all, logits_pc_img_mv_all]
        # pc / img means point cloud / image only inference
        log_suffix_list = ["_pc", "_img", "_merge", "_pc_img", "_img_mv", "_pc_img_mv"]

        for log_idx, logit in enumerate(logits_list):
            log_suffix = log_suffix_list[log_idx]
            for idx, labels in enumerate(labels_all):
                # labels / logit : [B, 1156]
                for i in torch.unique(labels):
                    label_idx = (labels == i)
                    if label_idx.sum() > 0:
                        logit_i = logit[idx]
                        per_cat_correct[i] += (logit_i[label_idx].argmax(dim=1) == labels[label_idx]).float().sum()
                        per_cat_count[i] += label_idx.sum()
            topk_acc, _ = self.accuracy(torch.cat(logit), torch.cat(labels_all), topk=(1, 3, 5,))
            per_cat_acc = per_cat_correct / per_cat_count
            logging.info('Test ObjaverseLVIS{4}: class_acc: {3:.2f} top1_acc: {0:.2f} top3_acc: {1:.2f} top5_acc: {2:.2f}'
                         .format(topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item(), per_cat_acc.mean()*100, log_suffix))

            wandb_folder = "test_lvis" + log_suffix # Eg: text_lvis_pc
            wandb.log({wandb_folder + "/epoch": self.epoch,
                       wandb_folder + "/step": self.step,
                       wandb_folder + "/class_acc" + str(log_suffix) + str(model_suffix): per_cat_acc.mean()*100,
                       wandb_folder + "/top1_acc" + str(log_suffix) + str(model_suffix): topk_acc[0],
                       wandb_folder + "/top3_acc" + str(log_suffix) + str(model_suffix): topk_acc[1],
                       wandb_folder + "/top5_acc" + str(log_suffix) + str(model_suffix): topk_acc[2],
                       })
            # Reset the counter for class average accuracy
            per_cat_correct, per_cat_count = torch.zeros(1156).to(self.config.device), torch.zeros(1156).to(
                self.config.device)
        """
        # Update the saving principle
        if overall_acc > self.best_lvis_acc:
            self.best_lvis_acc = overall_acc
            self.save_model('best_lvis')
        """

    def eval(self):
        self.test_modelnet40(self.model, "_sgd")
        self.test_objaverse_lvis(self.model, "_sgd")
        self.test_scanobjectnn(self.model, "_sgd")

    def save_model(self, name):
        torch.save({
            "state_dict": self.model.state_dict(),
            "state_dict_ema": self.model_ema.module.state_dict(),
            "text_proj": self.text_proj.state_dict() if self.config.training.use_text_proj else None,
            "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
            "pc_img_proj": self.pc_img_to_text_proj.state_dict() if self.config.training.pc_img_to_text else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.config.training.use_openclip_optimizer_scheduler == False else None,
            "epoch": self.epoch,
            "step": self.step,
            "image_logit_scale": self.image_logit_scale.state_dict(),
            "text_logit_scale": self.text_logit_scale.state_dict(),
            "best_img_contras_acc": self.best_img_contras_acc,
            "best_text_contras_acc": self.best_text_contras_acc,
            "best_modelnet40_overall_acc": self.best_modelnet40_overall_acc,
            "best_modelnet40_class_acc": self.best_modelnet40_class_acc,
            "best_lvis_acc": self.best_lvis_acc,
        }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res, correct

    def test_modelnet40(self, model, text_proj, log_suffix=""):
        model.eval()
        clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            text_proj.eval()
            clip_text_feat = text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(40).to(self.config.device)
        per_cat_count = torch.zeros(40).to(self.config.device)
        category2idx = self.modelnet40_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.modelnet40_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = model(data['xyz_dense'].cuda(), data['features_dense'].cuda())
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

                for i in range(40):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()
        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1, 3, 5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum()
        per_cat_acc = per_cat_correct / per_cat_count
        # for i in range(40):
        #    print(idx2category[i], per_cat_acc[i])

        if overall_acc > self.best_modelnet40_overall_acc:
            self.best_modelnet40_overall_acc = overall_acc
            self.save_model('best_modelnet40_overall')
        if per_cat_acc.mean() > self.best_modelnet40_class_acc:
            self.best_modelnet40_class_acc = per_cat_acc.mean()
            self.save_model('best_modelnet40_class')

        logging.info('Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(overall_acc,
                                                                                         self.best_modelnet40_overall_acc,
                                                                                         per_cat_acc.mean(),
                                                                                         self.best_modelnet40_class_acc))
        logging.info(
            'Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(), topk_acc[1].item(),
                                                                                topk_acc[2].item()))
        wandb.log({"test_modelnet40/epoch": self.epoch,
                   "test_modelnet40/step": self.step,
                   "test_modelnet40/overall_acc"+str(log_suffix): overall_acc,
                   "test_modelnet40/class_acc"+str(log_suffix): per_cat_acc.mean(),
                   "test_modelnet40/top3_acc"+str(log_suffix): topk_acc[1],
                   "test_modelnet40/top5_acc"+str(log_suffix): topk_acc[2], })

    def test_scanobjectnn(self, model, text_proj, log_suffix=""):
        model.eval()
        clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            text_proj.eval()
            clip_text_feat = text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(15).to(self.config.device)
        per_cat_count = torch.zeros(15).to(self.config.device)
        category2idx = self.scanobjectnn_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.scanobjectnn_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = model(data['xyz_dense'].cuda(), data['features_dense'].cuda())
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)
                # calculate per class accuracy
                for i in range(15):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()

        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1, 3, 5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum()
        per_cat_acc = per_cat_correct / per_cat_count

        logging.info('Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
        logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                           topk_acc[1].item(),
                                                                                           topk_acc[2].item()))
        wandb.log({"test_scanobjectnn/epoch": self.epoch,
                   "test_scanobjectnn/step": self.step,
                   "test_scanobjectnn/overall_acc"+str(log_suffix): overall_acc,
                   "test_scanobjectnn/class_acc"+str(log_suffix): per_cat_acc.mean(),
                   "test_scanobjectnn/top3_acc"+str(log_suffix): topk_acc[1],
                   "test_scanobjectnn/top5_acc"+str(log_suffix): topk_acc[2], })
