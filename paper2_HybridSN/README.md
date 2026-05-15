# HybridSN Reproduction for Hyperspectral Image Classification

This project is a reproduction implementation of **HybridSN** for hyperspectral image classification.  
The goal is to reproduce the main experimental pipeline of the original GitHub notebook and compare the reproduced results with the reported paper results.

The implementation follows the main HybridSN architecture:

- 3D convolution for joint spectral-spatial feature extraction
- 2D convolution for spatial feature refinement
- Fully connected layers for classification
- Softmax output for land-cover class prediction
- PCA-based spectral dimensionality reduction
- Patch-based training using hyperspectral image cubes

---

## 1. Environment

```text
Python: 3.9.8
TensorFlow: 2.10.0
Keras: 2.10.0
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

If GPU is used, make sure the TensorFlow version is compatible with your CUDA and cuDNN versions.

---

## 3. Dataset Preparation

Please place the dataset `.mat` files in the same directory as `HybridSN.py`.

Expected dataset files for Indian Pines:

```text
Indian_pines_corrected.mat
Indian_pines_gt.mat
```

The current implementation supports the following dataset format:

```text
Hyperspectral data: shape = (H, W, B)
Ground truth labels: shape = (H, W)
```

where:

- `H` is the image height.
- `W` is the image width.
- `B` is the number of spectral bands.
- Ground truth labels represent the class ID of each pixel.
- Label `0` is usually treated as background and removed before training.

---

## 4. How to Run

Run the main script:

```bash
python HybridSN.py
```

If using Jupyter Notebook:

```bash
jupyter notebook HybridSN_V1_New.ipynb
```

The script/notebook uses the following main dataset setting:

```python
dataset_name = "Indian Pines"
```

To change the dataset, modify the dataset loading section and update the corresponding `.mat` file paths.

---

## 5. Reference

```text
S. K. Roy, G. Krishna, S. R. Dubey, and B. B. Chaudhuri, "HybridSN: Exploring 3-D-2-D CNN Feature Hierarchy for Hyperspectral Image Classification," IEEE Geoscience and Remote Sensing Letters, vol. 17, no. 2, pp. 277-281, 2020, doi: 10.1109/LGRS.2019.2918719.
https://github.com/gokriznastic/HybridSN
```
