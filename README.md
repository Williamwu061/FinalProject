# CrossCUN Reproduction for Hyperspectral Unmixing

This project is a reproduction implementation of **CrossCUN** for hyperspectral image unmixing.  
The goal is to reproduce the main experimental pipeline of the original GitHub demo and compare the reproduced results with the reported paper results.

The implementation follows the main CrossCUN architecture:

- 3D convolution for spectral-spatial feature extraction
- 2D convolution for spatial feature refinement
- Softmax output for abundance estimation
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

Please place the dataset `.mat` files in the same directory as `CrossCUN.py`.

Expected dataset files:

```text
Samson.mat
JasperRidge.mat
TinyAPEX.mat
```

The current implementation supports the following dataset format:

```text
Y: hyperspectral data, shape = (L, N)
A: abundance ground truth, shape = (p, N)
H: image height
W: image width
L: number of spectral bands
p: number of endmembers
```

where:

- `L` is the number of spectral bands.
- `N = H × W` is the number of pixels.
- `p` is the number of endmembers.
- `A` is the ground-truth abundance matrix.

If the above format is not found, the script will try to use the original `helper.py` loading function for supported datasets.

---

## 4. How to Run

Run the main script:

```bash
python CrossCUN.py
```

The script will automatically run the following datasets:

```python
DATASET_CONFIGS = [
    {"dataset_name": "Samson", "mat_path": "Samson.mat"},
    {"dataset_name": "JasperRidge", "mat_path": "JasperRidge.mat"},
    {"dataset_name": "TinyAPEX", "mat_path": "TinyAPEX.mat"},
]
```

To run only one dataset, modify `DATASET_CONFIGS` in `CrossCUN.py`, for example:

```python
DATASET_CONFIGS = [
    {"dataset_name": "Samson", "mat_path": "Samson.mat"},
]
```

---

## 5. Reference

```text
X. Tao et al., "A New Deep Convolutional Network for Effective Hyperspectral Unmixing," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 15, pp. 6999-7012, 2022, doi: 10.1109/JSTARS.2022.3200733.
https://github.com/xuanwentao/IEEE_TGRS_CrossCUN.git
```
