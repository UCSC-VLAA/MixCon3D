import sys
import os
import logging
import shutil
import data
import models
import MinkowskiEngine as ME
import torch
import wandb
from omegaconf import OmegaConf
from datetime import datetime
from param import parse_args
from utils.misc import load_config, dump_config    
from utils.logger import setup_logging
from utils.scheduler import cosine_lr
from train import Trainer

from models.LogitScaleNetwork import LogitScaleNetwork
from models.mlp import Mlp

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from utils.model_ema import ModelEmaV2, SparseConvEmaV2
from utils.distributed import init_distributed_device

# Aim to solve "OSError: [Errno 24] Too many open files" problem
mp.set_sharing_strategy('file_system')


def cleanup():
    dist.destroy_process_group()


def main(args):
    cli_args, extras = parse_args(sys.argv[1:])

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config = load_config(cli_args.config, cli_args=vars(cli_args), extra_args=extras)
    device = init_distributed_device(args=config)
    # rank = torch.distributed.get_rank()

    if config.autoresume:
        config.trial_name = config.get('trial_name') + "@autoresume"
    else:
        config.trial_name = config.get('trial_name') + datetime.now().strftime('@%Y%m%d-%H%M%S')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')
    
    if config.rank == 0:
        os.makedirs(os.path.join(config.exp_dir, config.trial_name), exist_ok=config.autoresume)
        os.makedirs(config.ckpt_dir, exist_ok=True)
        if os.path.exists(config.code_dir):
            shutil.rmtree(config.code_dir)
        shutil.copytree("./src", config.code_dir)
    
    # config.device = 'cuda:{0}'.format(rank)
    if config.rank == 0:
        config.log_path = config.get('log_path') or os.path.join(config.exp_dir, config.trial_name, 'log.txt')
        config.log_level = logging.DEBUG if config.debug else logging.INFO
        setup_logging(config.log_path, config.log_level)
        dump_config(os.path.join(config.exp_dir, config.trial_name, 'config.yaml'), config)
        logging.info("Using {} GPU(s).".format(config.ngpu))
        wandb.init(reinit=True, project=config.project_name, name=config.trial_name, config=OmegaConf.to_object(config))

    if config.distributed:
        torch.distributed.barrier()

    # Write val code here
    if config.train:
        model = models.make(config).to(config.device)
        if config.rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            logging.info(model)
            logging.info("Network:{}, Number of parameters: {}".format(config.model.name, total_params))
        
        torch.cuda.set_device(config.rank)
        model.cuda(config.device)
        model.to(config.device)

        # Use separate logit_scale values
        image_logit_scale = LogitScaleNetwork(config.training.logit_scale_init).to(config.device)
        text_logit_scale = LogitScaleNetwork(config.training.logit_scale_init).to(config.device)
        img_text_logit_scale = LogitScaleNetwork(config.training.logit_scale_init).to(config.device)
        pc_img_to_text_logit_scale = LogitScaleNetwork(config.training.logit_scale_init).to(config.device)

        multi_view_proj = None
        # The use_MLP is set to False by default, the over-fitting is observed
        if config.training.use_MLP:
            image_proj = Mlp(in_features=config.model.out_channel).to(config.device)
            text_proj = Mlp(in_features=config.model.out_channel).to(config.device)
            pc_img_to_text_proj = Mlp(in_features=int(config.model.out_channel * 2),
                                      hidden_features=config.model.out_channel, out_features=config.model.out_channel).to(config.device)
            multi_view_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(config.device)
        else:
            image_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(config.device)
            text_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(config.device)
            pc_img_to_text_proj = torch.nn.Linear(int(config.model.out_channel * 2), config.model.out_channel).to(
                config.device)
            multi_view_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(config.device)

        # For EMA update, now support both the PointBERT and SparseConv for ema update
        model_ema = None
        image_proj_ema = None
        text_proj_ema = None
        pc_img_to_text_proj_ema = None
        multi_view_proj_ema = None
        if config.training.ema and not config.training.sparseconv_ema:
            model_ema = ModelEmaV2(model, decay=config.training.ema_decay, device=None)
            image_proj_ema = ModelEmaV2(image_proj, decay=config.training.ema_decay, device=None)
            text_proj_ema = ModelEmaV2(text_proj, decay=config.training.ema_decay, device=None)
            multi_view_proj_ema = ModelEmaV2(text_proj, decay=config.training.ema_decay, device=None)
            pc_img_to_text_proj_ema = ModelEmaV2(pc_img_to_text_proj, decay=config.training.ema_decay, device=None)

        # Due to the requirement of Minkowski, the ModelEmaV2 doesn't work on SparseConv.
        # So, I use the modified SparseConvEmaV2.
        if config.training.ema and config.training.sparseconv_ema:
            model_ema = SparseConvEmaV2(model, config=config, device=None)
            image_proj_ema = ModelEmaV2(image_proj, decay=config.training.ema_decay, device=None)
            text_proj_ema = ModelEmaV2(text_proj, decay=config.training.ema_decay, device=None)
            pc_img_to_text_proj_ema = ModelEmaV2(pc_img_to_text_proj, decay=config.training.ema_decay, device=None)

        model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=False)

        if config.model.name.startswith('Mink'):
            model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model) # minkowski only
            logging.info("Using MinkowskiSyncBatchNorm")
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logging.info("Using SyncBatchNorm")


        image_logit_scale = DDP(image_logit_scale, device_ids=[device], output_device=device, find_unused_parameters=False)
        text_logit_scale = DDP(text_logit_scale, device_ids=[device], output_device=device, find_unused_parameters=False)
        img_text_logit_scale = DDP(img_text_logit_scale, device_ids=[device], output_device=device, find_unused_parameters=False)


        pc_img_to_text_logit_scale = DDP(pc_img_to_text_logit_scale, device_ids=[device], output_device=device,
                                         find_unused_parameters=False)

        pc_img_to_text_proj = DDP(pc_img_to_text_proj, device_ids=[device], output_device=device,
                                  find_unused_parameters=False)

        image_proj = DDP(image_proj, device_ids=[device], output_device=device, find_unused_parameters=False)
        text_proj = DDP(text_proj, device_ids=[device], output_device=device, find_unused_parameters=False)
        multi_view_proj = DDP(multi_view_proj, device_ids=[device], output_device=device, find_unused_parameters=False)
        train_loader = data.make(config, 'train', config.rank, cli_args.ngpu)

        if config.rank == 0:
            modelnet40_loader = data.make_modelnet40test(config)
            objaverse_lvis_loader = data.make_objaverse_lvis(config)
            scanobjectnn_loader = data.make_scanobjectnntest(config)
        else:
            modelnet40_loader = None
            objaverse_lvis_loader = None
            scanobjectnn_loader = None

        if config.rank == 0:
            if train_loader is not None:
                logging.info("Train iterations: {}".format(len(train_loader) // config.dataset.accum_freq))
        if config.training.logit_scale_fix:
            params = list(model.parameters()) + list(image_proj.parameters()) + list(text_proj.parameters())
        else:
            params = list(model.parameters()) + list(image_proj.parameters()) + list(text_proj.parameters()) + \
                     list(text_logit_scale.parameters()) + list(image_logit_scale.parameters()) + \
                     list(img_text_logit_scale.parameters())
        if config.training.use_openclip_optimizer_scheduler:
            optimizer = torch.optim.AdamW(
                params,
                lr=config.training.lr,
                betas=(config.training.beta1, config.training.beta2),
                eps=config.training.eps,
            )
            #              steps_per_epoch * total_epoch
            total_steps = (len(train_loader) // config.dataset.accum_freq) * config.training.max_epoch
            warmup_steps = (len(train_loader) // config.dataset.accum_freq) * config.training.warmup_epoch
            logging.info("Total_Steps:{0} Warmup_Steps{1}".format(total_steps, warmup_steps))
            scheduler = cosine_lr(optimizer, config.training.lr, warmup_steps, total_steps, config.training.min_lr)
        else:
            optimizer = torch.optim.AdamW(params, lr=config.training.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.training.lr_decay_step, gamma=config.training.lr_decay_rate)

        trainer = Trainer(config.rank, config, model, text_logit_scale, image_logit_scale, img_text_logit_scale, pc_img_to_text_logit_scale,
                          image_proj, text_proj, pc_img_to_text_proj, multi_view_proj, optimizer, scheduler, train_loader, modelnet40_loader,
                          objaverse_lvis_loader, scanobjectnn_loader, model_ema, image_proj_ema, text_proj_ema,
                          pc_img_to_text_proj_ema, multi_view_proj_ema)

        if config.resume is not None:
            trainer.load_from_checkpoint(config.resume)
            if config.eval_only:
                trainer.eval()
                exit(0)
        elif config.autoresume:
            if os.path.exists(os.path.join(config.ckpt_dir, '{}.pt'.format('latest'))):
                trainer.load_from_checkpoint(os.path.join(config.ckpt_dir, '{}.pt'.format('latest')))

        trainer.train()

    # Yipeng: I notice that the wandb sometimes won't end and will upload files continuously
    # Currently, try to use reinit=True input parameter.
    if config.rank == 0:
        wandb.finish()
    cleanup()


if __name__ == '__main__':
    main(sys.argv[:1])
