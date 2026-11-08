"""
Selector adaptativo de estrategias v3.1 - ALWAYS INVESTED (Meb Faber style).

PRINCIPIO: el trading activo destruye valor vs SPY buy&hold.
Solucion: estar LONG siempre que precio > EMA200, salir solo en bear real.
Sector rotation: preferir sectores que superan al SPY.
"""
import numpy as np
import pandas as pd

from strategies.base import (
    momentum_signal, mean_reversion_signal, breakout_signal,
    trend_strength_signal, relative_strength_signal, dividend_capture_signal,
    always_invested_momentum, sector_rotation_signal,
)
from features.regime import detect_regime


def adaptive_signal(df_features, ml_probs, drawdown, config,
                     spy_returns=None, symbol=None):
    """
    Selector v3.1 - ALWAYS INVESTED.
    """
    reg = detect_regime(
        df_features,
        adx_thresh=config["regime"]["adx_trend_threshold"],
        vol_thresh=None,
    )

    aim = always_invested_momentum(df_features)
    sec_rot = sector_rotation_signal(df_features, spy_returns)
    mom = momentum_signal(df_features)
    mr = mean_reversion_signal(df_features)
    brk = breakout_signal(df_features, lookback=50)
    ts = trend_strength_signal(df_features)
    rs = relative_strength_signal(df_features, spy_returns) if spy_returns is not None else pd.Series(0, index=df_features.index)
    dc = dividend_capture_signal(df_features) if symbol in ("XLU", "XLP", "XLV", "XLFE") else pd.Series(0, index=df_features.index)
    ml_sig = ml_probs["ml_signal"] if "ml_signal" in ml_probs else pd.Series(0, index=df_features.index)

    above_ema200 = df_features["close"] > df_features["ema_200"]
    below_ema200 = df_features["close"] < df_features["ema_200"]
    for s in [mom, mr, brk, ts, rs, dc, ml_sig]:
        s.where(~((s < 0) & above_ema200), 0, inplace=True)
        s.where(~((s > 0) & below_ema200), 0, inplace=True)

    ml_high_conf = (ml_probs.get("prob_up", 0) > 0.60) | (ml_probs.get("prob_down", 0) > 0.60)
    ml_high_conf = ml_high_conf.reindex(df_features.index).fillna(False)

    signal = np.zeros(len(df_features), dtype=int)
    strategy = np.array(["NONE"] * len(df_features), dtype=object)
    mode = np.array(["NORMAL"] * len(df_features), dtype=object)

    recovery_thr = config["risk"]["recovery_threshold_pct"]
    max_dd = config["risk"]["max_drawdown_pct"]
    dd = drawdown.reindex(df_features.index).fillna(0).values

    for i in range(len(df_features)):
        cur_dd = dd[i]
        if cur_dd <= -max_dd:
            mode[i] = "CAPITAL_PRESERVATION"
            if sec_rot.iloc[i] != 0:
                signal[i] = int(sec_rot.iloc[i])
                strategy[i] = "SEC_ROT_HOLD"
            else:
                signal[i] = 0
                strategy[i] = "NONE_PRESERV"
            continue
        if cur_dd <= -recovery_thr:
            mode[i] = "RECOVERY"
        else:
            mode[i] = "NORMAL"

        r = reg.iloc[i]

        if mode[i] == "RECOVERY":
            if dc.iloc[i] != 0 and symbol in ("XLU", "XLP", "XLV", "XLFE"):
                signal[i] = int(dc.iloc[i])
                strategy[i] = "DIV_CAPTURE_RECOVERY"
            elif sec_rot.iloc[i] != 0:
                signal[i] = int(sec_rot.iloc[i])
                strategy[i] = "SEC_ROT_RECOVERY"
            elif aim.iloc[i] != 0:
                signal[i] = int(aim.iloc[i])
                strategy[i] = "ALWAYS_INV_RECOVERY"
            else:
                signal[i] = 0
                strategy[i] = "NONE_RECOVERY"
            continue

        if r in ("TREND_UP", "RANGE"):
            if sec_rot.iloc[i] != 0:
                signal[i] = int(sec_rot.iloc[i])
                strategy[i] = "SEC_ROTATION"
            elif ts.iloc[i] != 0:
                signal[i] = int(ts.iloc[i])
                strategy[i] = "TREND_STRENGTH"
            elif aim.iloc[i] != 0:
                signal[i] = int(aim.iloc[i])
                strategy[i] = "ALWAYS_INV"
            elif brk.iloc[i] != 0:
                signal[i] = int(brk.iloc[i])
                strategy[i] = "BREAKOUT"
            else:
                signal[i] = int(aim.iloc[i])
                strategy[i] = "ALWAYS_INV" if aim.iloc[i] != 0 else "NONE_RANGE"
        elif r == "TREND_DOWN":
            if above_ema200.iloc[i]:
                signal[i] = 1
                strategy[i] = "HOLD_ABOVE_EMA200"
            else:
                if dc.iloc[i] != 0 and symbol in ("XLU", "XLP", "XLV", "XLFE"):
                    signal[i] = int(dc.iloc[i])
                    strategy[i] = "DIV_CAPTURE_BEAR"
                else:
                    signal[i] = 0
                    strategy[i] = "NONE_BEAR"
        elif r == "HIGH_VOL":
            if dc.iloc[i] != 0 and symbol in ("XLU", "XLP", "XLV", "XLFE"):
                signal[i] = int(dc.iloc[i])
                strategy[i] = "DIV_CAPTURE_HIGHVOL"
            elif aim.iloc[i] != 0:
                signal[i] = int(aim.iloc[i])
                strategy[i] = "ALWAYS_INV_HIGHVOL"
            else:
                signal[i] = 0
                strategy[i] = "NONE_HIGHVOL"
        else:
            signal[i] = int(aim.iloc[i])
            strategy[i] = "ALWAYS_INV_DEFAULT" if aim.iloc[i] != 0 else "NONE"

    return pd.DataFrame(
        {"signal": signal, "strategy_used": strategy, "mode": mode, "regime": reg.values},
        index=df_features.index,
    )