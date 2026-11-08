"""
Modelo ML: LightGBM classifier para predecir direccion del precio a N barras.

Walk-forward training: en cada paso, entrena en los ultimos `train_window` barras
y predice las siguientes `test_window` barras. Evita lookahead y data leakage.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score

FEATURE_COLS = [
    "rsi_14", "rsi_7", "atr_pct", "adx_14",
    "bb_pct_b", "macd_hist",
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "vol_20", "vol_50",
    "price_above_ema50", "price_above_ema200",
    "ema9_above_ema21", "macd_hist_pos",
]


class MLModel:
    def __init__(self, params=None):
        self.params = params or {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "num_leaves": 15,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "verbosity": -1,
            "n_estimators": 100,
        }
        self.model = None
        self.label_map = {-1: 0, 0: 1, 1: 2}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

    def _align_labels(self, y):
        return y.map(self.label_map).values

    def fit(self, X, y):
        y_aligned = self._align_labels(y)
        mask = (~pd.Series(y_aligned).isna()) & X.notna().all(axis=1)
        Xc = X[mask].values
        yc = pd.Series(y_aligned)[mask].values
        if len(np.unique(yc)) < 2:
            self.model = None
            return self
        train_data = lgb.Dataset(Xc, label=yc, feature_name=list(X.columns))
        self.model = lgb.train(self.params, train_data, num_boost_round=self.params["n_estimators"])
        return self

    def predict_proba(self, X):
        if self.model is None:
            return pd.DataFrame(
                {"prob_down": 0.33, "prob_flat": 0.34, "prob_up": 0.33},
                index=X.index,
            )
        mask = X.notna().all(axis=1)
        probs = np.full((len(X), 3), np.nan)
        if mask.any():
            probs[mask.values] = self.model.predict(X[mask].values)
        return pd.DataFrame(
            {
                "prob_down": probs[:, 0] if self.model else 0.33,
                "prob_flat": probs[:, 1] if self.model else 0.34,
                "prob_up": probs[:, 2] if self.model else 0.33,
            },
            index=X.index,
        )

    def signal(self, X, threshold=0.50):
        probs = self.predict_proba(X)
        sig = np.where(probs["prob_up"] > threshold, 1,
                       np.where(probs["prob_down"] > threshold, -1, 0))
        return pd.Series(sig, index=X.index)

    def feature_importance(self, feature_names):
        if self.model is None:
            return pd.Series(dtype=float)
        imp = self.model.feature_importance(importance_type="gain")
        return pd.Series(imp, index=feature_names).sort_values(ascending=False)


def walk_forward_predict(df_features, label, train_window=252, test_window=63):
    """
    Walk-forward: en cada paso entrena en [t-train_window, t), predice [t, t+test_window).
    """
    feature_cols = FEATURE_COLS + ["regime_int"]
    X_all = df_features[feature_cols]
    y_all = label

    n = len(df_features)
    out_probs = pd.DataFrame(
        {"prob_down": np.nan, "prob_flat": np.nan, "prob_up": np.nan, "ml_signal": 0},
        index=df_features.index,
    )
    fold_metrics = []

    model = MLModel()
    start = train_window
    while start < n:
        end = min(start + test_window, n)
        train_idx = slice(max(0, start - train_window), start)
        test_idx = slice(start, end)

        X_train = X_all.iloc[train_idx]
        y_train = y_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        sig = model.signal(X_test, threshold=0.50)

        out_probs.loc[probs.index, "prob_down"] = probs["prob_down"].values
        out_probs.loc[probs.index, "prob_flat"] = probs["prob_flat"].values
        out_probs.loc[probs.index, "prob_up"] = probs["prob_up"].values
        out_probs.loc[sig.index, "ml_signal"] = sig.values

        y_test = y_all.iloc[test_idx]
        valid = (~y_test.isna()) & (sig != 0)
        if valid.sum() > 5:
            acc = float(accuracy_score(y_test[valid], sig[valid]))
        else:
            acc = 0.0
        fold_metrics.append({
            "start": str(X_test.index[0].date()),
            "end": str(X_test.index[-1].date()),
            "n_test": int(valid.sum()),
            "accuracy": acc,
        })

        start = end

    return out_probs, fold_metrics