import os
import json
import random
import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf

from helper import applyPCA, createImageCubes, splitTrainTestSet

try:
    from tensorflow.keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, Dropout, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import ModelCheckpoint
except Exception:
    from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, Dropout, Input
    from keras.models import Model
    from keras.callbacks import ModelCheckpoint


SEED = 42
WINDOW_SIZE = 9
PCA_DIM = 13
SPLIT_RATIO = 0.8
EPOCHS = 50
LR = 1e-4
DECAY = 1e-4
DROPOUT_RATE = 0.03
RESULT_DIR = "results"

DATASET_CONFIGS = [
    {"dataset_name": "Samson", "mat_path": "Samson.mat"},
    {"dataset_name": "JasperRidge", "mat_path": "JasperRidge.mat"},
    {"dataset_name": "TinyAPEX", "mat_path": "TinyAPEX.mat"},
]

os.makedirs(RESULT_DIR, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_adam(lr=LR, decay=DECAY):
    try:
        return tf.keras.optimizers.Adam(learning_rate=lr, decay=decay)
    except Exception:
        try:
            return tf.keras.optimizers.Adam(lr=lr, decay=decay)
        except Exception:
            print("[WARN] This Keras version does not support Adam decay; using Adam(learning_rate=lr).")
            return tf.keras.optimizers.Adam(learning_rate=lr)


def get_static_shape(x):
    if hasattr(x, "_keras_shape"):
        return x._keras_shape
    return tuple(x.shape.as_list())


def build_crosscun(window_size=WINDOW_SIZE, pca_dim=PCA_DIM, output_units=3):
    """Architecture aligned with demo.py."""
    input_layer = Input((window_size, window_size, pca_dim, 1))

    conv_layer1 = Conv3D(filters=128, kernel_size=(3, 3, 7), activation="tanh")(input_layer)
    conv_layer1 = Dropout(DROPOUT_RATE)(conv_layer1)

    conv_layer2 = Conv3D(filters=64, kernel_size=(3, 3, 5), activation="tanh")(conv_layer1)
    conv_layer2 = Dropout(DROPOUT_RATE)(conv_layer2)

    conv3d_shape = get_static_shape(conv_layer2)
    conv_layer2 = Reshape((
        conv3d_shape[1],
        conv3d_shape[2],
        conv3d_shape[3] * conv3d_shape[4]
    ))(conv_layer2)

    conv_layer2 = Conv2D(filters=32, kernel_size=(3, 3), activation="tanh")(conv_layer2)
    conv_layer3 = Dropout(DROPOUT_RATE)(conv_layer2)

    flatten_layer = Flatten()(conv_layer3)
    output_layer = Dense(units=output_units, activation="softmax")(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    adam = make_adam(LR, DECAY)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
    return model


def rmse_per_endmember(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))


def mean_rmse(y_true, y_pred):
    return float(np.mean(rmse_per_endmember(y_true, y_pred)))


def _scalar(value):
    return int(np.squeeze(value))


def load_testpy_format(mat_path):
    """
    Load original test.py format:
      Y: (L, N)
      A: (p, N)
      E: (L, p), optional for this script
      H, W, L, p: scalar

    Return:
      X_image: (H, W, L)
      y_image: (H, W, p)
      metadata dict
    """
    data = sio.loadmat(mat_path)
    required = ["Y", "A", "H", "W", "L", "p"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{mat_path} missing keys: {missing}")

    Y = data["Y"]
    A = data["A"]
    H = _scalar(data["H"])
    W = _scalar(data["W"])
    L = _scalar(data["L"])
    p = _scalar(data["p"])

    if Y.shape[0] != L:
        raise ValueError(f"Y first dimension {Y.shape[0]} != L {L}")
    if A.shape[0] != p:
        raise ValueError(f"A first dimension {A.shape[0]} != p {p}")
    if Y.shape[1] != H * W:
        raise ValueError(f"Y second dimension {Y.shape[1]} != H*W {H*W}")
    if A.shape[1] != H * W:
        raise ValueError(f"A second dimension {A.shape[1]} != H*W {H*W}")

    X_image = Y.T.reshape(H, W, L).astype(np.float32)
    y_image = A.T.reshape(H, W, p).astype(np.float32)

    return X_image, y_image, {
        "H": H,
        "W": W,
        "bands": L,
        "endmembers": p,
        "format": "test.py format: Y/A/H/W/L/p",
    }


def load_author_helper_fallback(dataset_name):
    """
    Optional fallback for author helper-style data.
    This keeps the original output dataset name, but internally maps only known names.
    """
    from helper import load

    helper_name_map = {
        "Samson": "Samson",
        "JasperRidge": "Jasper",  # helper.py uses 'Jasper'
    }

    if dataset_name not in helper_name_map:
        raise FileNotFoundError(
            f"No test.py-format MAT file found and no helper.py fallback exists for {dataset_name}."
        )

    X, y = load(helper_name_map[dataset_name])

    band = X.shape[0]
    endmember = y.shape[0]
    n_pixels = X.shape[1]
    side = int(np.sqrt(n_pixels))
    if side * side != n_pixels:
        raise ValueError(
            f"Author helper fallback assumes square image, but {dataset_name} has {n_pixels} pixels."
        )

    X_image = np.reshape(X, (band, side, side))
    X_image = np.transpose(X_image, (1, 2, 0)).astype(np.float32)

    y_image = np.reshape(y, (endmember, side, side))
    y_image = np.transpose(y_image, (1, 2, 0)).astype(np.float32)

    return X_image, y_image, {
        "H": side,
        "W": side,
        "bands": band,
        "endmembers": endmember,
        "format": f"helper.py fallback: load('{helper_name_map[dataset_name]}')",
    }


def load_dataset_with_original_name(dataset_name, mat_path):
    """
    Prefer original test.py MAT file in the current directory.
    If not found or not in the expected format, fallback to helper.py for Samson/JasperRidge.
    """
    candidate_paths = [
        mat_path,
        os.path.join(os.getcwd(), mat_path),
        os.path.join(os.getcwd(), "data", mat_path),
    ]

    existing_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            existing_path = p
            break

    if existing_path is not None:
        try:
            print(f"[INFO] Loading {dataset_name} from test.py-format file: {existing_path}")
            return load_testpy_format(existing_path)
        except KeyError as e:
            print(f"[WARN] {existing_path} is not test.py format: {e}")
            print("[WARN] Trying helper.py fallback if available...")

    print(f"[INFO] Trying helper.py fallback for {dataset_name}...")
    return load_author_helper_fallback(dataset_name)


def prepare_dataset(dataset_name, mat_path):
    X_image, y_image, meta = load_dataset_with_original_name(dataset_name, mat_path)

    # No extra normalization: align with demo.py.
    X_pca, pca = applyPCA(X_image, numComponents=PCA_DIM)
    X_patch, y_patch = createImageCubes(X_pca, y_image, windowSize=WINDOW_SIZE)

    Xtrain, ytrain, Xtest, ytest = splitTrainTestSet(X_patch, y_patch, SPLIT_RATIO)

    Xtrain = Xtrain.reshape(-1, WINDOW_SIZE, WINDOW_SIZE, PCA_DIM, 1)
    Xtest = Xtest.reshape(-1, WINDOW_SIZE, WINDOW_SIZE, PCA_DIM, 1)
    Xall = X_patch.reshape(-1, WINDOW_SIZE, WINDOW_SIZE, PCA_DIM, 1)

    return {
        "Xtrain": Xtrain,
        "ytrain": ytrain,
        "Xtest": Xtest,
        "ytest": ytest,
        "Xall": Xall,
        "yall": y_patch,
        "H": meta["H"],
        "W": meta["W"],
        "band": meta["bands"],
        "endmember": meta["endmembers"],
        "format": meta["format"],
        "pca": pca,
    }


def run_dataset(dataset_name, mat_path):
    print("\n" + "=" * 80)
    print(f"[RUN] {dataset_name}")
    print("=" * 80)

    ds = prepare_dataset(dataset_name, mat_path)

    print(f"[INFO] source format: {ds['format']}")
    print(f"[INFO] H={ds['H']}, W={ds['W']}, bands={ds['band']}, endmembers={ds['endmember']}")
    print(f"[INFO] Xtrain: {ds['Xtrain'].shape}, ytrain: {ds['ytrain'].shape}")
    print(f"[INFO] Xtest : {ds['Xtest'].shape}, ytest : {ds['ytest'].shape}")
    print(f"[INFO] Xall  : {ds['Xall'].shape}")

    model = build_crosscun(
        window_size=WINDOW_SIZE,
        pca_dim=PCA_DIM,
        output_units=ds["ytrain"].shape[1]
    )
    model.summary()

    weight_path = os.path.join(RESULT_DIR, f"{dataset_name}-best-model.weights.h5")

    checkpoint = ModelCheckpoint(
        weight_path,
        monitor="accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="max"
    )

    history = model.fit(
        x=ds["Xtrain"],
        y=ds["ytrain"],
        epochs=EPOCHS,
        callbacks=[checkpoint],
        verbose=1
    )

    if os.path.exists(weight_path):
        model.load_weights(weight_path)

    y_pred_test = model.predict(ds["Xtest"], verbose=0)
    test_rmse = rmse_per_endmember(ds["ytest"], y_pred_test)
    test_mean = mean_rmse(ds["ytest"], y_pred_test)

    print(f"[TEST] RMSE per endmember: {test_rmse}")
    print(f"[TEST] Mean RMSE         : {test_mean:.6f}")

    y_pred_all = model.predict(ds["Xall"], verbose=1)

    dataset_dir = os.path.join(RESULT_DIR, f"{dataset_name}result")
    os.makedirs(dataset_dir, exist_ok=True)
    prediction_txt = os.path.join(dataset_dir, "Y_pred_all.txt")
    np.savetxt(prediction_txt, y_pred_all, delimiter=",")

    result = {
        "dataset": dataset_name,
        "source_format": ds["format"],
        "H": int(ds["H"]),
        "W": int(ds["W"]),
        "bands": int(ds["band"]),
        "endmembers": int(ds["endmember"]),
        "window_size": WINDOW_SIZE,
        "pca_dim": PCA_DIM,
        "split_ratio_argument": SPLIT_RATIO,
        "note": "helper.py splitTrainTestSet uses this ratio as train size, despite the name test_ratio.",
        "train_samples": int(ds["Xtrain"].shape[0]),
        "test_samples": int(ds["Xtest"].shape[0]),
        "test_rmse_per_endmember": test_rmse.tolist(),
        "test_mean_rmse": float(test_mean),
        "best_weight_path": weight_path,
        "prediction_txt": prediction_txt,
    }

    with open(os.path.join(dataset_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main():
    set_seed(SEED)

    all_results = []
    for cfg in DATASET_CONFIGS:
        try:
            all_results.append(run_dataset(**cfg))
        except FileNotFoundError as e:
            print(f"[SKIP] {cfg['dataset_name']}: missing file -> {e}")
        except NotImplementedError as e:
            print(f"[SKIP] {cfg['dataset_name']}: {e}")
        except Exception as e:
            print(f"[ERROR] {cfg['dataset_name']}: {type(e).__name__}: {e}")
            raise

    if all_results:
        rows = []
        for r in all_results:
            row = {
                "Dataset": r["dataset"],
                "SourceFormat": r["source_format"],
                "H": r["H"],
                "W": r["W"],
                "Bands": r["bands"],
                "Endmembers": r["endmembers"],
                "TrainSamples": r["train_samples"],
                "TestSamples": r["test_samples"],
                "TestMeanRMSE": r["test_mean_rmse"],
            }
            for idx, value in enumerate(r["test_rmse_per_endmember"], start=1):
                row[f"RMSE_Endmember_{idx}"] = value
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(RESULT_DIR, "summary_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print("\n[SUMMARY]")
        print(df)
        print(f"Saved summary to: {csv_path}")


if __name__ == "__main__":
    main()
