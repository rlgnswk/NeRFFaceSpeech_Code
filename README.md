# Official NeRFFaceSpeech Code

## NeRFFaceSpeech: One-shot Audio-driven 3D Talking Head Synthesis via Generative Prior, [CVPR 2024 Workshop on AI for Content Creation (AI4CC)](https://ai4cc.net/)

[Paper](http://arxiv.org/abs/2405.05749/)  /  [Project Page](https://rlgnswk.github.io/NeRFFaceSpeech_ProjectPage/)


## Setting

We have confirmed that the code runs under the following conditions.

Python 3.7.16 // CUDA 11.7 //GPU 3090

```.bash
git clone https://github.com/rlgnswk/NeRFFaceSpeech_Code.git
cd NeRFFaceSpeech_Code/
conda env create -f environment.yml
conda activate nerffacespeech
```

#### Please install Nvdiffrast inside the Deep3DFaceRecon_pytorch folder.

```.bash
cd Deep3DFaceRecon_pytorch
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
```

## Download 

[Download Link](https://drive.google.com/drive/folders/1W3TGSh5ufmT3T1XPwU7LRB_y4bcbmm9i?usp=sharing)

```.bash
mkdir pretrained_networks
```


Download SadTalker_V0.0.2_256.safetensors
https://github.com/OpenTalker/SadTalker/releases to NeRFFaceSpeech_Code\pretrained_networks\sad_talker_pretrained

Download
https://huggingface.co/wsj1995/sadTalker/blob/af80749f8c9af3702fbd0272df14ff086986a1de/BFM09_model_info.mat to NeRFFaceSpeech_Code\pretrained_networks\BFM_for_3DMM-Fitting-Pytorch\BFM

@Thanks nitinmukesh's reports

### Place Pretrained Weights at pretrained_networks/

## Command (Generated from Latent Space)

```.bash

python StyleNeRF/main_NeRFFaceSpeech_audio_driven_from_z.py   \
    --outdir=out_test_z --trunc=0.7 \
        --network=pretrained_networks/ffhq_1024.pkl \
            --test_data="test_data/test_audio/AdamSchiff_0.wav" \
                --seeds=6;        

```

## Command (Generated from Real Image)

The inversion process for real image takes some time.

```.bash

python StyleNeRF/main_NeRFFaceSpeech_audio_driven_from_image.py   \
    --outdir=out_test_real --trunc=0.7 \
        --network=pretrained_networks/ffhq_1024.pkl \
            --test_data="test_data/test_audio/AdamSchiff_0.wav" \
                --test_img="test_data/test_img/32.png";       

```

## Command (Pose Varying)

The first command is for head pose varying only.

The second command is for head pose and exp varing by video-frames 
(at that time, audio input is only for the initial frame.)

The video frames should be pose-predictable.

```.bash

python StyleNeRF/main_NeRFFaceSpeech_audio_driven_w_given_poses.py   \
    --outdir=out_test_given_pose --trunc=0.7 \
        --network=pretrained_networks/ffhq_1024.pkl \
            --test_data="test_data/test_audio/AdamSchiff_0.wav" \
                --test_img="test_data/test_img/AustinScott0_0_cropped.jpg"\
                    --motion_guide_img_folder="driving_frames";     


python StyleNeRF/main_NeRFFaceSpeech_video_driven.py   \
    --outdir=out_test_video_driven --trunc=0.7 \
        --network=pretrained_networks/ffhq_1024.pkl \
            --test_data="test_data/test_audio/AdamSchiff_0.wav" \
                --test_img="test_data/test_img/DougJones_0_cropped.jpg"\
                    --motion_guide_img_folder="driving_frames";
```

## Custom Data for Use

### If you want to use new audio and image data, you must follow the formats of [StyleNeRF](https://github.com/facebookresearch/StyleNeRF) for image data and [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) or [SadTalker](https://github.com/OpenTalker/SadTalker) for audio data.

## Post-processing @ [nitinmukesh](https://github.com/nitinmukesh)

There is an applicable post-processing method called [GFPGAN](https://github.com/TencentARC/GFPGAN). It is being applied to other methods as well and can help produce better results. Please refer to the [issue](https://github.com/rlgnswk/NeRFFaceSpeech_Code/issues/5)!

## Caution: Error Accumulation

The proposed method may not work well due to accumulated errors such as landmark prediction errors and inversion(reconsturction) errors.

## Ethical Use

This project is intended for research and educational purposes only. Misuse of technology for deceptive practices is strictly discouraged

## Acknowledgement

We appreciate [StyleNeRF](https://github.com/facebookresearch/StyleNeRF), [PTI](https://github.com/danielroich/PTI), [Wav2Lip](https://github.com/Rudrabha/Wav2Lip), [SadTalker](https://github.com/OpenTalker/SadTalker), [Deep3Drecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch) and [3DMM-Fitting](https://github.com/ascust/3DMM-Fitting-Pytorch) for sharing their codes and baselines.

## Citation

```bibtex
@misc{kim2024nerffacespeech,
    title={NeRFFaceSpeech: One-shot Audio-driven 3D Talking Head Synthesis via Generative Prior}, 
    author={Gihoon Kim and Kwanggyoon Seo and Sihun Cha and Junyong Noh},
    year={2024},
    eprint={2405.05749},
    archivePrefix={arXiv},
    primaryClass={cs.CV}}
            
@misc{kim2024nerffacespeech,
    title={NeRFFaceSpeech: One-shot Audio-driven 3D Talking Head Synthesis via Generative Prior},
    author={Gihoon Kim, Kwanggyoon Seo, Sihun Cha and Junyong Noh},
    booktitle={IEEE Computer Vision and Pattern Recognition Workshops},
    year={2024}}
```
