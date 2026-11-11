import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score

FEATURES = ["vix","fear_greed","fed_funds","yield_spread","mom_3m","mom_6m","mom_12m","rsi","atr_pct","above_ema200"]

class MLPredictor:
    def __init__(self):
        self.model = None
        self.lookback = 252
        self.horizon = 21

    def prepare_features(self, prices, macro_features=None):
        df = prices.copy()
        df["mom_3m"] = df["close"].pct_change(63)
        df["mom_6m"] = df["close"].pct_change(126)
        df["mom_12m"] = df["close"].pct_change(252)
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"] = 100 - 100/(1+gain/loss.replace(0,np.nan))
        df["rsi"] = df["rsi"].fillna(50)
        hl = df["high"]-df["low"]
        df["atr_pct"] = (hl.rolling(14).mean()/df["close"]).fillna(0)
        df["above_ema200"] = (df["close"] > df["close"].rolling(200).mean()).astype(int)
        if macro_features is not None and not macro_features.empty:
            for col in ["vix","fear_greed","fed_funds","yield_spread"]:
                if col in macro_features.columns:
                    df[col] = macro_features[col].reindex(df.index).ffill().fillna(0)
        else:
            for col in ["vix","fear_greed","fed_funds","yield_spread"]:
                df[col] = 0
        return df

    def make_label(self, df):
        fwd = df["close"].shift(-self.horizon)/df["close"]-1
        std = fwd.std()
        thr = std*0.5 if std and std>0 else 0.01
        return pd.Series(np.where(fwd>thr,1,np.where(fwd<-thr,-1,0)),index=df.index)

    def train_predict(self, df):
        label = self.make_label(df)
        feat_df = df[FEATURES].copy()
        valid = label.notna() & feat_df.notna().all(axis=1)
        if valid.sum() < 100:
            return 0, 0.5
        X = feat_df[valid].values
        y = label[valid].values
        if len(np.unique(y))<2:
            return 0, 0.5
        split = int(len(X)*0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        if len(np.unique(y_tr))<2:
            return 0, 0.5
        tr_data = lgb.Dataset(X_tr,label=y_tr,feature_name=FEATURES)
        params = {"objective":"multiclass","num_class":3,"metric":"multi_logloss","num_leaves":15,"learning_rate":0.05,"verbosity":-1,"n_estimators":100}
        self.model = lgb.train(params,tr_data,num_boost_round=100)
        last = feat_df.iloc[-1:].values
        probs = self.model.predict(last)[0]
        pred = np.argmax(probs)-1
        acc = 0
        if len(y_te)>5 and len(np.unique(y_te))>1:
            pred_te = self.model.predict(X_te)
            pred_te = np.argmax(pred_te,axis=1)-1
            acc = accuracy_score(y_te,pred_te)
        return int(pred), float(acc)
