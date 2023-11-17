# guided-diffusion

This repository is based on codebase for [Diffusion Models Beat GANS on Image Synthesis](http://arxiv.org/abs/2105.05233) with repository [openai/guided-diffusion](https://github.com/openai/guided-diffusion) and [Semantic Hierarchy Emerges in Deep Generative Representations for Scene Synthesis](https://arxiv.org/pdf/1911.09267.pdf) with repository [genforce/higan](https://github.com/genforce/higan), aiming to apply interpretations on Diffusion models.

# Downloads
## Download pre-trained models

Before using these models, please review the corresponding [model card](model-card.md) to understand the intended use and limitations of these models.

Here are the download links for each model checkpoint:
The current model to be used is:
 * LSUN bedroom: [lsun_bedroom.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt)

Model could be used in future development is:

 * LSUN cat: [lsun_cat.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_cat.pt)
 * LSUN horse: [lsun_horse.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_horse.pt)
 * LSUN horse (no dropout): [lsun_horse_nodropout.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_horse_nodropout.pt)

We assume that you have downloaded the relevant model checkpoints into a folder called `pretrained/`.

## Download pretrained classifier

Gain classifier attributes weights from these links: [attributes] (https://github.com/kywch/Vis_places365/blob/master/W_sceneattribute_wideresnet18.npy) [weights] (http://places2.csail.mit.edu/models_places365/resnet152_places365.caffemodel)
Place your models under 'predictors/pretrain/'

## Download dataset

Use the script in `lsun/` to download whole dataset or certain-class dataset(recommended for our experiment):

```
python subdownload.py
```

# Diffusion models

## LSUN models

These models are class-unconditional and correspond to a single LSUN class. Here, we show how to sample from `lsun_bedroom.pt`, but the other two LSUN checkpoints should work as well:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python image_sample.py $MODEL_FLAGS --model_path models/lsun_bedroom.pt $SAMPLE_FLAGS
```

You can sample from `lsun_horse_nodropout.pt` by changing the dropout flag:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python image_sample.py $MODEL_FLAGS --model_path models/lsun_horse_nodropout.pt $SAMPLE_FLAGS
```

Note that for these models, the best samples result from using 1000 timesteps:

```
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 1000"
```

## For interpretation

For interpretation, you can quickly use following script to generate images:

```
python image_sample.py 
```

However, only simple image sampling is supported currently and the repo may needed batch sampling in the future.

# Classification model

This model's samples will be classied by a resnet in 'predictors/':

```
python scene_predictor.py
```

# Boundary model

The classification results and latent vectors from guided diffusion are obtained from previous sections. We need a boundary based on this 1-1 mapping, so use the script:

'''
python train_boundary.py
'''

to obtain a boundary for your interpretation.