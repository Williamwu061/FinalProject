# A2S2K-ResNet Reproduction for Hyperspectral Image Classification

This project is a reproduction implementation of **A2S2K-ResNet** for hyperspectral image classification.  
The goal is to reproduce the main experimental pipeline of the original GitHub code and compare the reproduced results with the reported paper results.

The implementation follows the main A2S2K-ResNet architecture:

- 3D convolution for spectral-spatial feature extraction
- Attention-based adaptive spectral-spatial kernel selection
- Improved 3D residual blocks for discriminative feature learning
- Efficient Feature Recalibration (EFR) / channel attention mechanism
- Softmax output for land-cover class prediction
- Patch-based training using hyperspectral image cubes

---

## 1. Environment

```text
Python: 3.9.8
PyTorch: 1.13.1+cu116
TorchSummary: 1.5.1
NumPy: 1.24.4
SciPy: 1.13.1
scikit-learn: 1.6.1
Pandas: 2.3.3
CUDA: 11.2
GPU: NVIDIA GeForce RTX 3060
OS: Windows-10-10.0.26200
```

---

## 2. Install required packages

```bash
pip install -r requirements.txt
```

If GPU is used, make sure the PyTorch version is compatible with your CUDA version.

---

## 3. Dataset Preparation

Please place the dataset `.mat` files in the same directory as `A2S2KResNet.py`.

Expected dataset files:

```text
Indian_pines_corrected.mat
Indian_pines_gt.mat
```

The current implementation supports the following datasets:

```text
IN  : Indian Pines
UP  : Pavia University
SV  : Salinas
KSC : Kennedy Space Center
```

The expected dataset format is:

```text
Hyperspectral data: shape = (H, W, B)
Ground truth labels: shape = (H, W)
```

where:

- `H` is the image height.
- `W` is the image width.
- `B` is the number of spectral bands.
- Ground truth labels represent the class ID of each pixel.
- Label `0` is treated as background and is not used for training/testing.

The code also saves outputs to the following folders:

```text
models/
report/
classification_maps/
```

---

## 4. How to Run

Run the main script:

```bash
python A2S2KResNet.py -d IN
```

Example with custom training settings:

```bash
python A2S2KResNet.py -d IN -e 200 -i 3 -p 4 -k 24 -vs 0.9 -o adam
```

The main arguments are:

```text
-d,  --dataset       Dataset name: IN, UP, SV, or KSC
-e,  --epoch         Number of training epochs
-i,  --iter          Number of repeated runs
-p,  --patch         Patch length; actual patch size = 2 × patch + 1
-k,  --kernel        Number of kernels
-vs, --valid_split   Split ratio argument
-o,  --optimizer     Optimizer name, such as adam or diffgrad
```

For example, when `-p 4` is used, the actual input patch size is:

```text
9 × 9
```

After training, numerical results can be found in:

```text
report/
```

Classification maps can be found in:

```text
classification_maps/
```

Saved model weights can be found in:

```text
models/
```

---

## 5. Reference

```text
S. K. Roy, S. Manna, T. Song, and L. Bruzzone, "Attention-Based Adaptive Spectral-Spatial Kernel ResNet for Hyperspectral Image Classification," IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 9, pp. 7831-7843, 2021, doi: 10.1109/TGRS.2020.3043267.
https://github.com/suvojit-0x55aa/A2S2K-ResNet
```
