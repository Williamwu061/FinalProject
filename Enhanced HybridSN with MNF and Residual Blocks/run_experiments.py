import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

if os.environ.get("HYBRIDSN_FORCE_CPU") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import keras
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from keras import layers, models
from keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras import optimizers

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


DATASETS = {
    "indian_pines": {
        "data_file": "Indian_pines_corrected.mat",
        "gt_file": "Indian_pines_gt.mat",
        "data_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt",
        "classes": [
            "Alfalfa",
            "Corn-notill",
            "Corn-mintill",
            "Corn",
            "Grass-pasture",
            "Grass-trees",
            "Grass-pasture-mowed",
            "Hay-windrowed",
            "Oats",
            "Soybean-notill",
            "Soybean-mintill",
            "Soybean-clean",
            "Wheat",
            "Woods",
            "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers",
        ],
    },
    "pavia_u": {
        "data_file": "PaviaU.mat",
        "gt_file": "PaviaU_gt.mat",
        "data_key": "paviaU",
        "gt_key": "paviaU_gt",
        "classes": None,
    },
    "salinas": {
        "data_file": "Salinas_corrected.mat",
        "gt_file": "Salinas_gt.mat",
        "data_key": "salinas_corrected",
        "gt_key": "salinas_gt",
        "classes": None,
    },
}


VARIANTS = {
    "baseline_pca": {"reduction": "pca", "residual": False},
    "residual_pca": {"reduction": "pca", "residual": True},
    "baseline_mnf": {"reduction": "mnf", "residual": False},
    "residual_mnf": {"reduction": "mnf", "residual": True},
}


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(data_root, dataset_name):
    cfg = DATASETS[dataset_name]
    data = sio.loadmat(data_root / cfg["data_file"])[cfg["data_key"]]
    gt = sio.loadmat(data_root / cfg["gt_file"])[cfg["gt_key"]]
    return data.astype(np.float32), gt.astype(np.int32), cfg


def apply_pca(cube, n_components, whiten=True, seed=0):
    h, w, bands = cube.shape
    flat = cube.reshape(-1, bands)
    pca = PCA(n_components=n_components, whiten=whiten, random_state=seed)
    reduced = pca.fit_transform(flat).reshape(h, w, n_components)
    info = {
        "method": "PCA",
        "n_components": int(n_components),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    return reduced.astype(np.float32), info


def estimate_noise_samples(cube):
    h_diff = cube[:, 1:, :] - cube[:, :-1, :]
    v_diff = cube[1:, :, :] - cube[:-1, :, :]
    noise = np.concatenate(
        [h_diff.reshape(-1, cube.shape[-1]), v_diff.reshape(-1, cube.shape[-1])],
        axis=0,
    )
    return noise / np.sqrt(2.0)


def apply_mnf(cube, n_components, regularization=1e-6):
    h, w, bands = cube.shape
    flat = cube.reshape(-1, bands).astype(np.float64)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean

    noise = estimate_noise_samples(cube).astype(np.float64)
    noise = noise - noise.mean(axis=0, keepdims=True)
    noise_cov = np.cov(noise, rowvar=False)
    signal_cov = np.cov(centered, rowvar=False)

    eps = regularization * max(float(np.trace(noise_cov)) / bands, 1.0)
    noise_cov = noise_cov + eps * np.eye(bands)

    noise_values, noise_vectors = np.linalg.eigh(noise_cov)
    noise_values = np.maximum(noise_values, eps)
    noise_whitener = noise_vectors @ np.diag(1.0 / np.sqrt(noise_values)) @ noise_vectors.T

    whitened_signal_cov = noise_whitener.T @ signal_cov @ noise_whitener
    mnf_values, mnf_vectors = np.linalg.eigh(whitened_signal_cov)
    order = np.argsort(mnf_values)[::-1]
    selected = order[:n_components]
    transform = noise_whitener @ mnf_vectors[:, selected]

    reduced = centered @ transform
    reduced = (reduced - reduced.mean(axis=0, keepdims=True)) / (
        reduced.std(axis=0, keepdims=True) + 1e-8
    )
    info = {
        "method": "MNF",
        "n_components": int(n_components),
        "regularization": float(regularization),
        "mnf_eigenvalues": mnf_values[selected].tolist(),
        "noise_cov_trace": float(np.trace(noise_cov)),
    }
    return reduced.reshape(h, w, n_components).astype(np.float32), info


def pad_with_zeros(cube, margin):
    padded = np.zeros(
        (cube.shape[0] + 2 * margin, cube.shape[1] + 2 * margin, cube.shape[2]),
        dtype=np.float32,
    )
    padded[margin : margin + cube.shape[0], margin : margin + cube.shape[1], :] = cube
    return padded


def select_labeled_coords(labels, max_samples_per_class=None, seed=0):
    coords = []
    rng = np.random.default_rng(seed)
    for cls in sorted(int(v) for v in np.unique(labels) if v > 0):
        cls_coords = np.argwhere(labels == cls)
        if max_samples_per_class and len(cls_coords) > max_samples_per_class:
            selected = rng.choice(len(cls_coords), size=max_samples_per_class, replace=False)
            cls_coords = cls_coords[selected]
        coords.extend(cls_coords.tolist())
    return np.array(coords, dtype=np.int32)


def create_image_cubes(cube, labels, window_size, coords=None):
    margin = (window_size - 1) // 2
    padded = pad_with_zeros(cube, margin)
    if coords is not None:
        patches = np.empty((len(coords), window_size, window_size, cube.shape[2]), dtype=np.float32)
        y = np.empty((len(coords),), dtype=np.int32)
        for idx, (r, c) in enumerate(coords):
            pr = r + margin
            pc = c + margin
            patches[idx] = padded[pr - margin : pr + margin + 1, pc - margin : pc + margin + 1, :]
            y[idx] = labels[r, c] - 1
        return patches, y, coords.astype(np.int32)

    # Match HybridSN_V1_New.ipynb: build every spatial patch in row-major order,
    # then remove label 0 and shift labels to 0-based class ids.
    patches = np.zeros((cube.shape[0] * cube.shape[1], window_size, window_size, cube.shape[2]))
    patch_labels = np.zeros((cube.shape[0] * cube.shape[1]))
    patch_coords = np.zeros((cube.shape[0] * cube.shape[1], 2), dtype=np.int32)
    patch_index = 0
    for r in range(margin, padded.shape[0] - margin):
        for c in range(margin, padded.shape[1] - margin):
            patch = padded[r - margin : r + margin + 1, c - margin : c + margin + 1]
            patches[patch_index, :, :, :] = patch
            patch_labels[patch_index] = labels[r - margin, c - margin]
            patch_coords[patch_index] = [r - margin, c - margin]
            patch_index += 1

    keep = patch_labels > 0
    patches = patches[keep, :, :, :].astype(np.float32)
    patch_labels = (patch_labels[keep] - 1).astype(np.int32)
    patch_coords = patch_coords[keep]
    return patches, patch_labels, patch_coords


def subset_by_class(x, y, coords, max_samples_per_class, seed):
    if not max_samples_per_class:
        return x, y, coords
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(y):
        indices = np.where(y == cls)[0]
        if len(indices) > max_samples_per_class:
            indices = rng.choice(indices, size=max_samples_per_class, replace=False)
        keep.extend(indices.tolist())
    keep = np.array(sorted(keep), dtype=np.int64)
    return x[keep], y[keep], coords[keep]


def split_indices(y, test_ratio, val_ratio, seed):
    all_indices = np.arange(len(y))
    try:
        train_val_idx, test_idx = train_test_split(
            all_indices,
            test_size=test_ratio,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        train_val_idx, test_idx = train_test_split(
            all_indices,
            test_size=test_ratio,
            random_state=seed,
            stratify=None,
        )

    if val_ratio > 0:
        try:
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio,
                random_state=seed,
                stratify=y[train_val_idx],
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio,
                random_state=seed,
                stratify=None,
            )
    else:
        train_idx = train_val_idx
        val_idx = np.array([], dtype=np.int64)

    return train_idx, val_idx, test_idx


def residual_3d_block(x, filters, name):
    shortcut = x
    y = layers.Conv3D(filters, kernel_size=(3, 3, 3), padding="same", activation="relu", name=f"{name}_conv1")(x)
    y = layers.Conv3D(filters, kernel_size=(3, 3, 3), padding="same", activation=None, name=f"{name}_conv2")(y)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv3D(filters, kernel_size=(1, 1, 1), padding="same", activation=None, name=f"{name}_proj")(shortcut)
    y = layers.Add(name=f"{name}_add")([shortcut, y])
    return layers.Activation("relu", name=f"{name}_relu")(y)


def residual_2d_block(x, filters, name):
    shortcut = x
    y = layers.Conv2D(filters, kernel_size=(3, 3), padding="same", activation="relu", name=f"{name}_conv1")(x)
    y = layers.Conv2D(filters, kernel_size=(3, 3), padding="same", activation=None, name=f"{name}_conv2")(y)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding="same", activation=None, name=f"{name}_proj")(shortcut)
    y = layers.Add(name=f"{name}_add")([shortcut, y])
    return layers.Activation("relu", name=f"{name}_relu")(y)


def build_hybridsn(input_shape, num_classes, residual=False):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv3D(filters=8, kernel_size=(3, 3, 7), activation="relu", name="conv3d_1")(inputs)
    if residual:
        x = residual_3d_block(x, 8, "res3d_1")

    x = layers.Conv3D(filters=16, kernel_size=(3, 3, 5), activation="relu", name="conv3d_2")(x)
    if residual:
        x = residual_3d_block(x, 16, "res3d_2")

    x = layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation="relu", name="conv3d_3")(x)
    if residual:
        x = residual_3d_block(x, 32, "res3d_3")

    shape = x.shape
    x = layers.Reshape((int(shape[1]), int(shape[2]), int(shape[3]) * int(shape[4])), name="reshape_3d_to_2d")(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", name="conv2d_1")(x)
    if residual:
        x = residual_2d_block(x, 64, "res2d_1")

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(units=256, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(units=128, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.4, name="dropout_2")(x)
    outputs = layers.Dense(units=num_classes, activation="softmax", name="classifier")(x)

    return models.Model(inputs=inputs, outputs=outputs)


def per_class_accuracy(cm):
    denominator = cm.sum(axis=1)
    return np.divide(
        np.diag(cm),
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )


def save_history_plot(history_csv, output_path):
    hist = pd.read_csv(history_csv)
    plt.figure(figsize=(7, 4))
    if "accuracy" in hist:
        plt.plot(hist["epoch"], hist["accuracy"], label="train_acc")
    if "val_accuracy" in hist:
        plt.plot(hist["epoch"], hist["val_accuracy"], label="val_acc")
    if "loss" in hist:
        plt.plot(hist["epoch"], hist["loss"], label="train_loss")
    if "val_loss" in hist:
        plt.plot(hist["epoch"], hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_confusion_matrix_plot(cm, output_path, class_names=None):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    if class_names and len(class_names) == cm.shape[0]:
        plt.xticks(ticks, class_names, rotation=90, fontsize=6)
        plt.yticks(ticks, class_names, fontsize=6)
    else:
        plt.xticks(ticks, ticks + 1)
        plt.yticks(ticks, ticks + 1)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_one_variant(args, variant_name, variant_cfg, seed, run_root):
    set_seed(seed)
    data_root = Path(args.data_root).resolve()
    run_dir = run_root / f"{variant_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cube, gt, dataset_cfg = load_dataset(data_root, args.dataset)
    print(f"[INFO] Loaded dataset={args.dataset}, cube={cube.shape}, gt={gt.shape}")
    if variant_cfg["reduction"] == "pca":
        reduced, reduction_info = apply_pca(cube, args.components, whiten=True, seed=seed)
    elif variant_cfg["reduction"] == "mnf":
        reduced, reduction_info = apply_mnf(cube, args.components, args.mnf_regularization)
    else:
        raise ValueError(f"Unknown reduction: {variant_cfg['reduction']}")
    print(f"[INFO] Reduction={reduction_info['method']}, reduced_cube={reduced.shape}")

    x, y, coords = create_image_cubes(reduced, gt, args.window_size)
    x, y, coords = subset_by_class(x, y, coords, args.max_samples_per_class, seed)
    x = np.expand_dims(x, axis=-1)
    num_classes = int(np.max(y)) + 1

    train_idx, val_idx, test_idx = split_indices(y, args.test_ratio, args.val_ratio, seed)
    print(
        f"[INFO] Samples: total={len(y)}, train={len(train_idx)}, "
        f"val={len(val_idx)}, test={len(test_idx)}, classes={num_classes}"
    )
    np.savez(
        run_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        coords=coords,
        labels=y,
    )

    y_cat = keras.utils.to_categorical(y, num_classes=num_classes)
    model = build_hybridsn(
        input_shape=(args.window_size, args.window_size, args.components, 1),
        num_classes=num_classes,
        residual=variant_cfg["residual"],
    )
    if args.lr_schedule == "exponential":
        learning_rate = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=args.decay_steps,
            decay_rate=args.decay_rate,
        )
    else:
        learning_rate = args.learning_rate

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    model_summary = []
    model.summary(print_fn=model_summary.append)
    (run_dir / "model_summary.txt").write_text("\n".join(model_summary), encoding="utf-8")

    config = {
        "variant": variant_name,
        "seed": int(seed),
        "dataset": args.dataset,
        "data_root": str(data_root),
        "reduction": reduction_info,
        "residual": bool(variant_cfg["residual"]),
        "window_size": int(args.window_size),
        "components": int(args.components),
        "test_ratio": float(args.test_ratio),
        "val_ratio": float(args.val_ratio),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "lr_schedule": args.lr_schedule,
        "decay_steps": int(args.decay_steps),
        "decay_rate": float(args.decay_rate),
        "max_samples_per_class": args.max_samples_per_class,
        "samples_total": int(len(y)),
        "samples_train": int(len(train_idx)),
        "samples_val": int(len(val_idx)),
        "samples_test": int(len(test_idx)),
        "model_params": int(model.count_params()),
    }
    write_json(run_dir / "config.json", config)

    callbacks = [CSVLogger(run_dir / "history.csv")]
    if len(val_idx):
        callbacks.append(
            ModelCheckpoint(
                filepath=run_dir / "best_model.keras",
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1,
            )
        )

    validation_data = None
    if len(val_idx):
        validation_data = (x[val_idx], y_cat[val_idx])

    start = time.time()
    print(
        f"[INFO] Start training: epochs={args.epochs}, batch_size={args.batch_size}, "
        f"validation={'yes' if len(val_idx) else 'no'}, selected_model="
        f"{'best validation checkpoint' if len(val_idx) else 'final epoch'}"
    )
    history = model.fit(
        x[train_idx],
        y_cat[train_idx],
        validation_data=validation_data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=args.verbose,
    )
    train_time = time.time() - start

    final_model_path = run_dir / "final_model.keras"
    model.save(final_model_path)

    best_model_path = run_dir / "best_model.keras"
    if len(val_idx) and best_model_path.exists():
        model = tf.keras.models.load_model(best_model_path)

    test_loss, test_acc = model.evaluate(x[test_idx], y_cat[test_idx], batch_size=args.batch_size, verbose=0)
    y_prob = model.predict(x[test_idx], batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = y[test_idx]

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    each_acc = per_class_accuracy(cm)
    oa = accuracy_score(y_true, y_pred)
    aa = float(np.mean(each_acc))
    kappa = cohen_kappa_score(y_true, y_pred)

    class_names = dataset_cfg["classes"]
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names if class_names and len(class_names) == num_classes else None,
        digits=4,
        zero_division=0,
    )

    (run_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    np.save(run_dir / "confusion_matrix.npy", cm)
    pd.DataFrame(cm).to_csv(run_dir / "confusion_matrix.csv", index=False)
    np.save(run_dir / "test_predictions.npy", y_pred)
    np.save(run_dir / "test_probabilities.npy", y_prob)
    save_history_plot(run_dir / "history.csv", run_dir / "training_curve.png")
    save_confusion_matrix_plot(cm, run_dir / "confusion_matrix.png", class_names)

    metrics = {
        "variant": variant_name,
        "seed": int(seed),
        "oa": float(oa),
        "aa": float(aa),
        "kappa": float(kappa),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "train_time_sec": float(train_time),
        "model_params": int(model.count_params()),
        "per_class_accuracy": each_acc.tolist(),
        "selected_model": "best_validation" if len(val_idx) and best_model_path.exists() else "final_epoch",
        "best_epoch": int(np.argmax(history.history.get("val_accuracy", history.history.get("accuracy"))) + 1),
    }
    write_json(run_dir / "metrics.json", metrics)
    return {**config, **metrics, "run_dir": str(run_dir)}


def summarize(results, run_root):
    df = pd.DataFrame(results)
    df.to_csv(run_root / "summary_results.csv", index=False)

    agg = (
        df.groupby("variant")
        .agg(
            oa_mean=("oa", "mean"),
            oa_std=("oa", "std"),
            aa_mean=("aa", "mean"),
            aa_std=("aa", "std"),
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
            test_loss_mean=("test_loss", "mean"),
            params=("model_params", "first"),
            runs=("seed", "count"),
        )
        .reset_index()
    )
    agg.to_csv(run_root / "summary_by_variant.csv", index=False)

    lines = [
        "# Ablation Summary",
        "",
        "| Variant | OA mean | OA std | AA mean | AA std | Kappa mean | Kappa std | Runs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.oa_mean:.4f} | {0.0 if pd.isna(row.oa_std) else row.oa_std:.4f} | "
            f"{row.aa_mean:.4f} | {0.0 if pd.isna(row.aa_std) else row.aa_std:.4f} | "
            f"{row.kappa_mean:.4f} | {0.0 if pd.isna(row.kappa_std) else row.kappa_std:.4f} | {row.runs} |"
        )
    (run_root / "ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HybridSN ablation experiments with PCA/MNF and residual modules."
    )
    parser.add_argument("--data-root", default="data", help="Folder containing .mat datasets.")
    parser.add_argument("--dataset", default="indian_pines", choices=sorted(DATASETS.keys()))
    parser.add_argument(
        "--variants",
        default="baseline_pca,residual_pca,baseline_mnf,residual_mnf",
        help="Comma-separated variants.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[345], help="Random seeds.")
    parser.add_argument("--components", type=int, default=30, help="Number of PCA/MNF components.")
    parser.add_argument("--window-size", type=int, default=25, help="Spatial patch size.")
    parser.add_argument("--test-ratio", type=float, default=0.7, help="Test split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.0, help="Validation ratio inside train split.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--lr-schedule", choices=["exponential", "constant"], default="exponential")
    parser.add_argument("--decay-steps", type=int, default=10000)
    parser.add_argument("--decay-rate", type=float, default=0.9)
    parser.add_argument("--mnf-regularization", type=float, default=1e-6)
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Optional small-sample mode for quick smoke tests.",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    selected_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for variant in selected_variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {', '.join(VARIANTS)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.dataset}_{timestamp}"
    run_root = Path(args.output_dir) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    write_json(run_root / "experiment_config.json", vars(args))
    results = []
    for variant in selected_variants:
        for seed in args.seeds:
            print(f"\n[RUN] variant={variant}, seed={seed}")
            result = run_one_variant(args, variant, VARIANTS[variant], seed, run_root)
            results.append(result)
            summarize(results, run_root)

    print(f"\n[DONE] Results saved to: {run_root.resolve()}")
    print(f"[DONE] Summary CSV: {(run_root / 'summary_by_variant.csv').resolve()}")


if __name__ == "__main__":
    main()
