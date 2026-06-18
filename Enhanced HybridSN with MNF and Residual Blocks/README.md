# Enhanced HybridSN with MNF and Residual Blocks

本專案是 CSIE2103 類神經網路期末論文實作，基於 HybridSN 進行改良與消融實驗。  
主要比較原始 `PCA + HybridSN`，以及加入 `MNF` 降維與 `Residual Block` 後的效果。

## 實驗方法

本實驗包含四組 ablation：

| Variant | 降維方法 | Residual Block |
|---|---|---|
| `baseline_pca` | PCA | 無 |
| `residual_pca` | PCA | 有 |
| `baseline_mnf` | MNF | 無 |
| `residual_mnf` | MNF | 有 |

主要評估指標：

- Overall Accuracy (OA)
- Average Accuracy (AA)
- Kappa coefficient
- Confusion Matrix
- Per-class Accuracy

## 環境需求

```text
Python: 3.9.8
TensorFlow: 2.10.0
Keras: 2.10.0
NumPy: 1.24.4
SciPy: 1.13.1
scikit-learn: 1.6.1
Pandas: 2.3.3
CUDA: 11.2
GPU: NVIDIA GeForce RTX 5080
```

安裝套件：

```bash
pip install -r requirements.txt
```

## 正式實驗

正式實驗建議跑四組方法，並使用多個 random seed：

```bash
python run_experiments.py --epochs 100 --seeds 345 346 347 --variants baseline_pca,residual_pca,baseline_mnf,residual_mnf --run-name formal_indian_pines
```

預設設定：

| 參數 | 預設值 |
|---|---:|
| Dataset | Indian Pines |
| Data folder | `data/` |
| Window size | `25` |
| PCA/MNF components | `30` |
| Test ratio | `0.7` |
| Validation ratio | `0.0` |
| Batch size | `256` |
| Learning rate | `0.001` |
| LR schedule | ExponentialDecay, decay_steps=`10000`, decay_rate=`0.9` |
| Epochs | `100` |

## 輸出結果

執行後，結果會自動儲存在：

```text
results/<run_name>/
```

每一組實驗會有自己的資料夾，例如：

```text
results/formal_indian_pines/baseline_pca_seed345/
```

主要輸出檔案：

| 檔案 | 說明 |
|---|---|
| `metrics.json` | OA、AA、Kappa、test loss、per-class accuracy |
| `history.csv` | 每個 epoch 的 loss / accuracy |
| `training_curve.png` | 訓練曲線圖 |
| `confusion_matrix.csv` | confusion matrix 表格 |
| `confusion_matrix.png` | confusion matrix 圖 |
| `classification_report.txt` | precision、recall、F1-score |
| `model_summary.txt` | 模型架構與參數量 |
| `split_indices.npz` | train / validation / test index 記錄 |
| `final_model.keras` | 最後一個 epoch 的模型 |
| `best_model.keras` | 有設定 validation 時，依 validation accuracy 儲存的最佳模型 |

總表輸出：

| 檔案 | 說明 |
|---|---|
| `summary_results.csv` | 每個 seed / variant 的完整結果 |
| `summary_by_variant.csv` | 各 variant 的 mean / std 統計 |
| `ablation_summary.md` | 可轉成 LaTeX 的 ablation table |
| `experiment_config.json` | 本次實驗參數設定 |

## Reference

```text
S. K. Roy, G. Krishna, S. R. Dubey, and B. B. Chaudhuri, "HybridSN: Exploring 3-D-2-D CNN Feature Hierarchy for Hyperspectral Image Classification," IEEE Geoscience and Remote Sensing Letters, vol. 17, no. 2, pp. 277-281, 2020, doi: 10.1109/LGRS.2019.2918719.
https://github.com/gokriznastic/HybridSN
```
