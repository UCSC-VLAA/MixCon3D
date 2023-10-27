# MixCon3D
The code for MixCon3D

## Installation

## Data Downloading

## Training
To train the PointBERT model using the MixCon3D, using the following command:

```python
torchrun --nproc_per_node=4 src/main.py --ngpu 4 dataset.folder=data5 dataset.train_batch_size=180 dataset.image_feat_mode=Multiple dataset.inference_image_feat_mode=Multiple dataset.image_amount=4 dataset.accum_freq=1 dataset.num_workers=3 model.name=PointBERT model.scaling=3 model.use_dense=True training.use_text_proj=True training.use_image_proj=True training.image_text_align=True training.share_proj=True training.pc_img_to_text=True training.logit_scale_fix=False training.lr=0.006 training.min_lr=3e-3 training.max_epoch=200 training.debug=True --trial_name MixCon3D_clean_code_sc3_mv4_feat-avg_shr-proj --config src/configs/train.yaml
```
