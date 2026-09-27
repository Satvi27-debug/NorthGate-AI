"""PRD Section 9 - Deep Learning Architecture.

Import order note
-----------------
TensorFlow is imported **before** anything that pulls in pyarrow. On this
environment, ``import pyarrow.compute`` (which ``pandas`` loads lazily for
parquet) followed by ``import tensorflow`` aborts with a DLL initialisation
failure inside ``_pywrap_tensorflow_internal`` - a native-library conflict
between the two packages, reproducible across all permutations tested. Importing
TensorFlow first, and only then pandas, works reliably, and parquet read/write
still behaves normally afterwards. ``src/common`` imports no arrow-backed
readers, so importing this module first is sufficient.


Four genuine sequence architectures consuming the ``(samples, L, n_features)``
windows of Listing 5.2:

    LSTM            LSTM -> Dropout -> LSTM -> Dense
    GRU             GRU  -> Dropout -> GRU  -> Dense
    Bidirectional   BiLSTM -> Dropout -> Dense
    Transformer     Positional enc. -> Multi-head attention -> FFN -> Dense

Training discipline (Section 9.4): chronological 70/15/15, ``shuffle=False``,
dropout, early stopping with ``restore_best_weights``, Huber loss to temper
return spikes, and every random seed fixed.

Boundary safety (Section 5.3)
-----------------------------
The split is computed on the **date axis** first, and windows are then built
*inside each partition independently*. A window therefore cannot contain a
single row from another partition, which is the leak the PRD warns about.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# TensorFlow MUST be imported before pandas/pyarrow on this environment. See the
# module docstring: the reverse order aborts with a DLL initialisation failure
# inside _pywrap_tensorflow_internal. Importing it here, first, makes that
# impossible for this entry point.
import tensorflow as tf  # noqa: E402  isort:skip

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    DL_METRICS,
    DL_PREDICTIONS,
    FEATURES,
    MODEL_LEADERBOARD,
    MODELS_DIR,
    PROCESSED_DIR,
    banner,
    ensure_dirs,
    get_logger,
    load_config,
    set_seed,
    write_json,
)
from src.features import feature_columns, make_sequences, split_by_date  # noqa: E402
from src.models_ml import compute_metrics  # noqa: E402

LOG = get_logger("models_dl")


# ==========================================================================
# Architectures (Section 9.1)
# ==========================================================================
def _optimizer(lr: float):
    import tensorflow as tf

    return tf.keras.optimizers.Adam(lr)


def build_lstm(L, n_features, units, dropout, lr, loss):
    """LSTM -> Dropout -> LSTM -> Dropout -> Dense -> regression head (Listing 9.1)."""
    from tensorflow.keras import layers, models

    m = models.Sequential([
        layers.Input((L, n_features)),
        layers.LSTM(units[0], return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units[1]),
        layers.Dropout(dropout),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name="lstm")
    m.compile(optimizer=_optimizer(lr), loss=loss, metrics=["mae"])
    return m


def build_gru(L, n_features, units, dropout, lr, loss):
    """GRU -> Dropout -> GRU -> Dense: fewer gates than LSTM, same interface."""
    from tensorflow.keras import layers, models

    m = models.Sequential([
        layers.Input((L, n_features)),
        layers.GRU(units[0], return_sequences=True),
        layers.Dropout(dropout),
        layers.GRU(units[1]),
        layers.Dropout(dropout),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name="gru")
    m.compile(optimizer=_optimizer(lr), loss=loss, metrics=["mae"])
    return m


def build_bilstm(L, n_features, units, dropout, lr, loss):
    """BiLSTM -> Dropout -> Dense: reads the window forward and backward."""
    from tensorflow.keras import layers, models

    m = models.Sequential([
        layers.Input((L, n_features)),
        layers.Bidirectional(layers.LSTM(units[0])),
        layers.Dropout(dropout),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name="bilstm")
    m.compile(optimizer=_optimizer(lr), loss=loss, metrics=["mae"])
    return m


def _sinusoidal_positional_encoding(d_model: int):
    """Sinusoidal positional encoding (Listing 9.2), as a Keras layer."""
    from tensorflow.keras import layers

    class SinusoidalPositionalEncoding(layers.Layer):
        def __init__(self, d_model, **kwargs):
            super().__init__(**kwargs)
            self.d_model = d_model

        def call(self, x):
            import tensorflow as tf

            length = tf.shape(x)[1]
            pos = tf.cast(tf.range(length)[:, None], tf.float32)
            i = tf.cast(tf.range(self.d_model)[None, :], tf.float32)
            angle = pos / tf.pow(
                10000.0, (2.0 * tf.floor(i / 2.0)) / tf.cast(self.d_model, tf.float32)
            )
            pe = tf.where(tf.cast(i, tf.int32) % 2 == 0, tf.sin(angle), tf.cos(angle))
            return x + pe[None, :, :]

    return SinusoidalPositionalEncoding(d_model)


def build_transformer(L, n_features, dcfg, dropout, lr, loss):
    """Encoder-only Transformer: positional enc. -> MHA -> FFN, stacked."""
    from tensorflow.keras import layers, models

    tcfg = dcfg["transformer"]
    d_model, heads, ff, blocks = tcfg["key_dim"], tcfg["heads"], tcfg["ff_dim"], tcfg["blocks"]

    def block(x):
        attn = layers.MultiHeadAttention(heads, d_model, dropout=dropout)(x, x)
        x = layers.LayerNormalization()(x + attn)          # residual + norm
        f = layers.Dense(ff, activation="relu")(x)
        f = layers.Dense(x.shape[-1])(f)
        return layers.LayerNormalization()(x + f)          # residual + norm

    inp = layers.Input((L, n_features), name="input")
    x = layers.Dense(d_model, name="embed")(inp)
    x = _sinusoidal_positional_encoding(d_model)(x)
    for _ in range(blocks):
        x = block(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1, name="regression_head")(x)

    m = models.Model(inp, out, name="transformer")
    m.compile(optimizer=_optimizer(lr), loss=loss, metrics=["mae"])
    return m


BUILDERS = {
    "LSTM": lambda L, f, c, d, lr, ls: build_lstm(L, f, c["units_lstm"], d, lr, ls),
    "GRU": lambda L, f, c, d, lr, ls: build_gru(L, f, c["units_gru"], d, lr, ls),
    "BiLSTM": lambda L, f, c, d, lr, ls: build_bilstm(L, f, c["units_bilstm"], d, lr, ls),
    "Transformer": lambda L, f, c, d, lr, ls: build_transformer(L, f, c, d, lr, ls),
}


# ==========================================================================
# Data preparation
# ==========================================================================
def prepare_sequences(df: pd.DataFrame, cols: list[str], cfg: dict) -> dict:
    """Build (X, y) window sets per chronological partition.

    Windows are constructed **inside each partition**, so a training window can
    never contain a validation or test row (Section 5.3).

    ``dl.per_ticker_windows`` selects how rows are ordered inside a partition:

    * **True (default)** - windows are built within each ticker separately, so a
      window is ``sequence_length`` consecutive sessions of one asset. This is
      the correct semantics for a sequence model: a window interleaving AAPL,
      MSFT and JPM rows would describe no real market path.
    * **False** - a single global ordering by (Date, Ticker), which yields more
      windows but mixes assets inside each one.

    Either way the scaler is fit on the training partition only (Section 5.4).
    """
    from sklearn.preprocessing import StandardScaler

    L = int(cfg["sequence_length"])
    horizon = int(cfg["target"]["horizon"])
    per_ticker = bool(cfg["dl"].get("per_ticker_windows", True))
    bounds = split_by_date(df["Date"], cfg)

    train_rows = df[df["Date"].isin(bounds["train"])]
    scaler = StandardScaler().fit(train_rows[cols].to_numpy(dtype=float))
    target_mu = float(train_rows["Target"].mean())
    target_sd = float(train_rows["Target"].std(ddof=0)) or 1.0

    out: dict = {}
    for name, dates in bounds.items():
        part = df[df["Date"].isin(dates)].sort_values(["Date", "Ticker"]).reset_index(drop=True)
        X_all = scaler.transform(part[cols].to_numpy(dtype=float))
        y_all = (part["Target"].to_numpy(dtype=float) - target_mu) / target_sd

        Xs, ys, keep_idx = [], [], []
        if per_ticker:
            for _, idx in part.groupby("Ticker", sort=True).groups.items():
                pos = np.sort(part.index.get_indexer(idx))
                w, t = make_sequences(X_all[pos], y_all[pos], L, horizon)
                if len(w) == 0:
                    continue
                Xs.append(w)
                ys.append(t)
                # Row p+L-1 is the last session inside the window; the target is
                # the session immediately after it.
                keep_idx.append(pos[L - 1: L - 1 + len(t)])
        else:
            w, t = make_sequences(X_all, y_all, L, horizon)
            Xs.append(w)
            ys.append(t)
            keep_idx.append(np.arange(L - 1, L - 1 + len(t)))

        if not Xs:
            raise RuntimeError(
                f"partition {name!r} produced no windows; sequence_length={L} may exceed "
                "the available history for at least one ticker"
            )

        flat_idx = np.concatenate(keep_idx)
        order = np.argsort(flat_idx, kind="stable")
        Xcat = np.concatenate(Xs, axis=0)[order]
        ycat = np.concatenate(ys, axis=0)[order]

        out[name] = {
            "X": np.ascontiguousarray(Xcat, dtype="float32"),
            "y": np.ascontiguousarray(ycat, dtype="float32"),
            "rows": part.iloc[flat_idx[order]].reset_index(drop=True),
            "n_rows": len(part),
            "n_windows": len(Xcat),
            "dates": dates,
        }
        LOG.info("  %-5s %6d rows -> %6d windows %s | %s -> %s",
                 name, len(part), len(Xcat), str(Xcat.shape[1:]),
                 pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date())

    out["_scaler"] = {"feature": scaler, "target_mu": target_mu, "target_sd": target_sd,
                      "columns": cols, "seq_length": L, "horizon": horizon,
                      "per_ticker": per_ticker}
    return out


# ==========================================================================
# Training
# ==========================================================================
def train_one(name: str, data: dict, cfg: dict, seed: int) -> dict:
    import tensorflow as tf
    from tensorflow.keras import callbacks

    dcfg = cfg["dl"]
    L = int(cfg["sequence_length"])
    n_features = data["train"]["X"].shape[2]
    dropout = float(dcfg["transformer"]["dropout"]) if name == "Transformer" else 0.2

    tf.keras.backend.clear_session()
    set_seed(seed)

    model = BUILDERS[name](L, n_features, dcfg, dropout,
                           float(dcfg["learning_rate"]), dcfg["loss"])

    es = callbacks.EarlyStopping(
        monitor="val_loss", patience=int(dcfg["patience"]),
        restore_best_weights=True, mode="min",
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=max(2, int(dcfg["patience"]) // 2),
        min_lr=1e-6, mode="min",
    )

    t0 = time.time()
    history = model.fit(
        data["train"]["X"], data["train"]["y"],
        validation_data=(data["val"]["X"], data["val"]["y"]),
        epochs=int(dcfg["epochs"]),
        batch_size=int(dcfg["batch_size"]),
        callbacks=[es, reduce_lr],
        shuffle=False,                     # Section 9.2 / 9.4
        verbose=0,
    )
    elapsed = time.time() - t0

    train_loss = list(history.history["loss"])
    val_loss = list(history.history.get("val_loss", []))
    epochs_run = len(train_loss)
    gap = float(val_loss[-1] - train_loss[-1]) if val_loss else float("nan")
    diagnosis = (
        "overfitting" if np.isfinite(gap) and gap > 0.05
        else "underfitting" if train_loss and train_loss[-1] > 0.5
        else "healthy"
    )

    LOG.info("  %-12s params=%7d epochs=%3d/%3d  train=%.5f val=%.5f  gap=%+.5f [%s]  %.1fs",
             name, model.count_params(), epochs_run, int(dcfg["epochs"]),
             train_loss[-1], val_loss[-1] if val_loss else float("nan"),
             gap, diagnosis, elapsed)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / f"{name.lower()}.keras")

    curves = pd.DataFrame({
        "epoch": range(1, epochs_run + 1),
        "train_loss": train_loss,
        "val_loss": val_loss,
    })
    curves.insert(0, "model", name)
    curves.to_csv(PROCESSED_DIR / f"loss_curve_{name.lower()}.csv", index=False)

    return {
        "model": model, "params": int(model.count_params()),
        "epochs_run": epochs_run, "train_loss": float(train_loss[-1]),
        "val_loss": float(val_loss[-1]) if val_loss else None,
        "overfit_gap": gap, "diagnosis": diagnosis,
        "elapsed": round(elapsed, 1), "curves": curves,
    }


# ==========================================================================
# Main
# ==========================================================================
def run_dl_models() -> dict:
    ensure_dirs()
    cfg = load_config()
    seed = set_seed(cfg.get("random_seed", 42))
    t0 = time.time()

    banner(LOG, "PRD Section 9 - Deep Learning Architecture")
    if not FEATURES.exists():
        raise FileNotFoundError(f"{FEATURES} not found. Run `python src/features.py` first.")

    df = pd.read_parquet(FEATURES)
    df["Date"] = pd.to_datetime(df["Date"])
    cols = feature_columns(df)
    LOG.info("Feature table: %d rows x %d features", len(df), len(cols))

    LOG.info("Building sequences (L=%d, horizon=%d, per-ticker=%s) inside each partition:",
             int(cfg["sequence_length"]), int(cfg["target"]["horizon"]),
             bool(cfg["dl"].get("per_ticker_windows", True)))
    data = prepare_sequences(df, cols, cfg)
    # `train_one` reads the sequence length from cfg, so the private scaler entry
    # is consumed here and never passed to a model trainer.
    scaler_info = data.pop("_scaler")
    tm, tsd = scaler_info["target_mu"], scaler_info["target_sd"]
    L = int(cfg["sequence_length"])

    rows, preds, curves = [], [], []
    for name in ("LSTM", "GRU", "BiLSTM", "Transformer"):
        LOG.info("")
        LOG.info("Training %s:", name)
        info = train_one(name, data, cfg, seed)

        y_true = data["test"]["y"] * tsd + tm
        y_pred = info["model"].predict(data["test"]["X"], verbose=0).ravel() * tsd + tm

        aligned = data["test"]["rows"].reset_index(drop=True)
        if len(aligned) != len(y_true):
            raise RuntimeError(
                f"prediction/row misalignment for {name}: {len(y_true)} predictions "
                f"vs {len(aligned)} rows"
            )

        price = aligned["Adjusted Close"].to_numpy(dtype=float)
        m = compute_metrics(y_true, y_pred, price)
        m.update({
            "Model": name, "Family": "DL", "Params": info["params"],
            "Epochs": info["epochs_run"], "Train_Loss": info["train_loss"],
            "Val_Loss": info["val_loss"], "Overfit_Gap": info["overfit_gap"],
            "Diagnosis": info["diagnosis"], "Train_Seconds": info["elapsed"],
            "Best_Params": json.dumps({"L": L, "n_features": len(cols),
                                       "loss": cfg["dl"]["loss"]}),
        })
        rows.append(m)
        LOG.info("  -> TEST  RMSE=%.6f MAE=%.6f MAPE=%.4f%% R2=%7.4f DirAcc=%.2f%%",
                 m["RMSE"], m["MAE"], m["MAPE_Price"], m["R2"], 100 * m["DirAcc"])

        block = aligned[["Date", "Ticker", "Adjusted Close", "Target"]].copy()
        block["Actual_Return"] = y_true
        block["Predicted_Return"] = y_pred
        block["Residual"] = y_true - y_pred
        block["Predicted_Price"] = price * np.exp(y_pred)
        block["Actual_Price"] = price * np.exp(y_true)
        block["Model"] = name
        preds.append(block)
        curves.append(info["curves"])

    pd.concat(preds, ignore_index=True).to_parquet(DL_PREDICTIONS, index=False)
    pd.concat(curves, ignore_index=True).to_csv(PROCESSED_DIR / "dl_loss_curves.csv", index=False)
    pd.DataFrame(rows).to_csv(DL_METRICS, index=False)

    # -- Section 9.5 headline table, combined with ML ----------------------
    ml_path = PROCESSED_DIR / "ml_metrics.csv"
    base_path = PROCESSED_DIR / "baseline_metrics.csv"
    if ml_path.exists() and base_path.exists():
        combined = pd.concat([pd.read_csv(base_path), pd.read_csv(ml_path),
                              pd.DataFrame(rows)], ignore_index=True)
        keep = [c for c in ["Model", "Family", "RMSE", "MAE", "MAPE", "R2", "DirAcc", "N"]
                if c in combined.columns]
        combined = combined[keep].sort_values("MAE").reset_index(drop=True)
        combined.to_csv(MODEL_LEADERBOARD, index=False)

        LOG.info("")
        banner(LOG, "PRD Section 9.5 - Eight-model comparison (one held-out test window)")
        head = f"{'Model':<22}{'RMSE':>11}{'MAE':>11}{'MAPE%':>9}{'R2':>9}{'DirAcc':>10}"
        LOG.info(head)
        LOG.info("-" * len(head))
        for _, r in combined.iterrows():
            LOG.info(f"{r['Model']:<22}{r['RMSE']:>11.6f}{r['MAE']:>11.6f}"
                     f"{r['MAPE']:>9.3f}{r['R2']:>9.4f}{100 * r['DirAcc']:>9.2f}%")
        LOG.info("-" * len(head))

    summary = {
        "generated_by": "src/models_dl.py",
        "seed": seed,
        "sequence_length": L,
        "per_ticker_windows": bool(cfg["dl"].get("per_ticker_windows", True)),
        "n_features": len(cols),
        "test_windows": int(len(data["test"]["y"])),
        "models": [{k: v for k, v in r.items() if k != "Best_Params"} for r in rows],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    write_json(summary, PROCESSED_DIR / "dl_run_summary.json")
    LOG.info("")
    LOG.info("Metrics     -> %s", DL_METRICS)
    LOG.info("Predictions -> %s", DL_PREDICTIONS)
    LOG.info("Loss curves -> %s", PROCESSED_DIR / "dl_loss_curves.csv")
    LOG.info("Elapsed %.1fs", summary["elapsed_seconds"])
    return summary


if __name__ == "__main__":
    run_dl_models()
