# BVSR-IK

The code of the paper "Blind Video Super-Resolution based on Implicit Kernels".

# Requirements

CUDA==11.6 Python==3.7 Pytorch==1.13

## Environment
```python
conda create -n BVSR python=3.7 -y && conda activate BVSR

git clone --depth=1 https://github.com/QZ1-boy/BVSR && cd QZ1-boy/BVSR/

# given CUDA 11.6
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

python -m pip install tqdm lmdb pyyaml opencv-python scikit-image
```

## Datasets Download
Training GT Datasets:

[REDS dataset] [REDS-GT](https://seungjunnah.github.io/Datasets/reds.html)

Testing GT Datasets:

[REDS4 dataset] [REDS4-GT](https://seungjunnah.github.io/Datasets/reds.html), [Vid4 dataset] [Vid4-GT](https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA), [UDM10 dataset] [UDM10-GT](https://github.com/psychopa4/PFNL)

Testing Datasets on Gaussian Blur:

[REDS4-Gaussian Blur](https://seungjunnah.github.io/Datasets/reds.html),
[Vid4-Gaussian Blur](https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA),
[UDM10-Gaussian Blur](https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA)

Testing Datasets on Realistic Motion Blur:

[REDS4-Realistic Motion Blur](https://seungjunnah.github.io/Datasets/reds.html),
[Vid4-Realistic Motion Blur](https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA),
[UDM10-Realistic Motion Blur](https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA)


# Train
```python
python train_Gaussian.py
python train_Realistic.py
```
# Test
```python
python test_Gaussian_REDS4.py
python test_Gaussian_Vid4.py
python test_Gaussian_UDM10.py 
```
# Citation
If this repository is helpful to your research, please cite our paper:
```python
@article{zhu2025blind,
  title={Blind Video Super-Resolution based on Implicit Kernels},
  author={Zhu, Qiang and Jiang, Yuxuan and Zhu, Shuyuan and Zhang, Fan and Bull, David and Zeng, Bing},
  journal={arXiv preprint arXiv:2503.07856},
  year={2025}
}
```
