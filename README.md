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

## Dataset Download
Training Datasets:

[REDS dataset] [REDS-GT](https://seungjunnah.github.io/Datasets/reds.html)

Testing GT Datasets:

[REDS4 dataset] [REDS4-GT](https://seungjunnah.github.io/Datasets/reds.html), [Vid4 dataset] [Vid4-GT](https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA), [UDM10 dataset] [UDM10-GT](https://github.com/psychopa4/PFNL)

Testing Datasets on Gaussian Blur:

[REDS4-Gaussian Blur](https://seungjunnah.github.io/Datasets/reds.html),
[Vid4-Gaussian Blur](https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA),
[UDM10-Gaussian Blur](https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA)

Testing Datasets on Realistic Motion Blur:

[REDS4-Realistic Motion Blur] (https://seungjunnah.github.io/Datasets/reds.html)

[Vid4-Realistic Motion Blur] (https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA)

[UDM10-Realistic Motion Blur] (https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA)

## Generation of Low-Resolution Compressed Videos
The LDB coding configuration (HM16.25) is adopted to compress the low-resolution videos downsampled by Bicubic. 

You can also obtain these testing datasets from our Google Drive. But training low-resolution compressed videos should be re-generated by the released codec. 


# Train
```python
python train_LD_37.py
```
# Test
```python
python test_LD_37.py 
```
# Citation
If this repository is helpful to your research, please cite our paper:
```python
@article{zhu2025fcvsr,
  title={FCVSR: A Frequency-aware Method for Compressed Video Super-Resolution},
  author={Zhu, Qiang and Zhang, Fan and Chen, Feiyu and Zhu, Shuyuan and Bull, David and Zeng, Bing},
  journal={arXiv preprint arXiv:2502.06431},
  year={2025}
}
```
