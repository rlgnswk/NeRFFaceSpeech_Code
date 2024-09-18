# Official NeRFFaceSpeech Code

## NeRFFaceSpeech: One-shot Audio-driven 3D Talking Head Synthesis via Generative Prior, CVPRW 2024

[Paper](http://arxiv.org/abs/2405.05749/)

[Project Page](https://rlgnswk.github.io/NeRFFaceSpeech_ProjectPage/)

## Setting

```.bash
git clone https://github.com/rlgnswk/NeRFFaceSpeech_Code.git
cd NeRFFaceSpeech_Code/
conda env create -f environment.yml
conda activate nerffacespeech
```

## Download Pretrained Weights in pretrained_networks/

[Download Link](https://drive.google.com/drive/folders/1W3TGSh5ufmT3T1XPwU7LRB_y4bcbmm9i?usp=sharing)

```.bash
mkdir pretrained_networks
```

### Place Pretrained Weights at pretrained_networks/

## Nvdiffrast 

### Please install Nvdiffrast inside the Deep3DFaceRecon_pytorch folder.

```.bash
cd Deep3DFaceRecon_pytorch
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
```

## Command (Generated from latent space)

```.bash

python StyleNeRF/main_NeRFFaceSpeech_audio_driven_from_z.py   \
    --outdir=out_test --trunc=0.7 \
        --network=pretrained_networks/ffhq_1024.pkl \
            --test_data="test_data/test_audio/AdamSchiff_0.wav" \
                --seeds=0;        

```

## Acknowledgement

We appreciate [StyleNeRF](https://github.com/facebookresearch/StyleNeRF), [PTI](https://github.com/danielroich/PTI), [Wav2Lip](https://github.com/Rudrabha/Wav2Lip), [SadTalker](https://github.com/OpenTalker/SadTalker), [Deep3Drecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch) and [3DMM-Fitting](https://github.com/ascust/3DMM-Fitting-Pytorch) for sharing their codes and baselines.

