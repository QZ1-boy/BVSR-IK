# BVSR-IK

The code of the paper "Blind Video Super-Resolution based on Implicit Kernels".

# Requirements

Python 3.9, PyTorch >= 1.9.1

Platforms: Ubuntu 22.04

## Environment
```python
conda create -n BVSR python=3.9 -y && conda activate BVSR

git clone --depth=1 https://github.com/QZ1-boy/BVSR && cd QZ1-boy/BVSR/

# given CUDA 11.1
python -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install tqdm lmdb pyyaml opencv-python scikit-image
```

## Datasets Download
Training GT Datasets:

[REDS dataset] [REDS-GT](https://seungjunnah.github.io/Datasets/reds.html)

Testing GT Datasets:

[REDS4 dataset] [REDS4-GT](https://seungjunnah.github.io/Datasets/reds.html), [Vid4 dataset] [Vid4-GT](https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA), [UDM10 dataset] [UDM10-GT](https://github.com/psychopa4/PFNL)

Testing Datasets on Gaussian Blur and Realistic Motion Blur:

[REDS4/Vid4/UDM10](https://pan.baidu.com/s/1u2rVDD7wfhpMGByKuSGr9w), Code [BVSR].

Put the downloaded training datasets and testing datasets into the ./dataset file path. 


# Train
```python
CUDA_VISIBLE_DEVICES=1   python main_KCA.py  --train --Deg_option Gaussian_REDS --config_path exp_KCA_REDS_Gaussian.cfg 
CUDA_VISIBLE_DEVICES=1   python main_KCA.py  --train --Deg_option Realistic_REDS --config_path exp_KCA_REDS_Realistic.cfg 
```
# Test
```python
CUDA_VISIBLE_DEVICES=1   python main_KCA.py  --test_REDS4  --Deg_option Gaussian_REDS  --config_path exp_KCA_REDS_Gaussian.cfg
CUDA_VISIBLE_DEVICES=1   python main_KCA.py  --test_Vid4   --Deg_option Gaussian_REDS  --config_path exp_KCA_REDS_Gaussian.cfg 
CUDA_VISIBLE_DEVICES=1   python main_KCA.py  --test_UDM10  --Deg_option Gaussian_REDS  --config_path exp_KCA_REDS_Gaussian.cfg
CUDA_VISIBLE_DEVICES=1   python main_KCA.py  --test_REDS4  --Deg_option Realistic_REDS  --config_path exp_KCA_REDS_Realistic.cfg
CUDA_VISIBLE_DEVICES=1   python main_KCA.py  --test_Vid4   --Deg_option Realistic_REDS  --config_path exp_KCA_REDS_Realistic.cfg 
CUDA_VISIBLE_DEVICES=1   python main_KCA.py  --test_UDM10  --Deg_option Realistic_REDS  --config_path exp_KCA_REDS_Realistic.cfg
```
# Citation
If this repository is helpful to your research, please cite our paper:
```python
@article{zhu2025blind,
  title={Blind Video Super-Resolution based on Implicit Kernels},
  author={Zhu, Qiang and Jiang, Yuxuan and Zhu, Shuyuan and Zhang, Fan and Bull, David and Zeng, Bing},
  conference={International Conference on Computer Vision},
  year={2025}
}
```
# Related Works
Our project was built on the video super-resolution method [FMA-Net](https://github.com/KAIST-VICLab/FMA-Net). We also release some blind video super-resolution models, e.g., [DBVSR](https://github.com/csbhr/Deep-Blind-VSR), [BSVSR](https://github.com/XY-boy/Blind-Satellite-VSR), [Self-BVSR](https://github.com/csbhr/Self-Blind-VSR).
