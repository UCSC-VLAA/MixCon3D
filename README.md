# MixCon3D: Synergizing Multi-View and Cross-Modal Contrastive Learning for Enhancing 3D Representation
This repository contains the official implementation of "MixCon3D" in our paper.

<p align="center">
  <img src="./figs/mixcon3d.jpg" width="1080">
Overview of the MixCon3D. We integrate the image and point cloud modality information, formulating a holistic 3D instance-level representation for cross-modal alignment.
</p>

## Installation
Please refer to this [instruction](https://github.com/UCSC-VLAA/MixCon3D/blob/main/Installation.md) for step-by-step installation guidance. Both the necessary packages and some helpful debug experience are provided.
## Data Downloading
First, modify the path in the download_data.py.
Then, execute the following command to download data from Hugging Face:
```python
python3 download_data.py
```
The datasets used for experiments are the same as [OpenShape](https://github.com/Colin97/OpenShape_code).
Please refer to OpenShape for more details of the data.

## Training
To train the PointBERT model using the MixCon3D, use the following command:

```python
torchrun --nproc_per_node=4 src/main.py --ngpu 4 dataset.folder=data5 dataset.train_batch_size=180 dataset.image_feat_mode=Multiple dataset.inference_image_feat_mode=Multiple dataset.image_amount=4 dataset.accum_freq=1 dataset.num_workers=3 model.name=PointBERT model.scaling=3 model.use_dense=True training.use_text_proj=True training.use_image_proj=True training.image_text_align=True training.share_proj=True training.pc_img_to_text=True training.logit_scale_fix=False training.lr=0.006 training.min_lr=3e-3 training.max_epoch=200 training.debug=True --trial_name MixCon3D_clean_code_sc3_mv4_feat-avg_shr-proj --config src/configs/train.yaml
```


## Acknowledgment
This codebase is based on [OpenShape](https://github.com/Colin97/OpenShape_code). Thanks to the authors for their awesome contributions! This work is partially supported by TPU Research Cloud (TRC) program and Google Cloud Research Credits program.

## Contact 
Questions and discussions are welcome via [gaoyp23@mail2.sysu.edu.cn](gaoyp23@mail2.sysu.edu.cn) or open an issue here.
