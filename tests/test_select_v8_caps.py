import os, sys
sys.path.insert(0, os.path.abspath("."))
from trading_bot.backtest.run_v8_hedge_fund import (
    load_macro_features, build_momentum_table, get_all_etf_scores, select_v8, UNIVERSE_V8,
)

def test_select_v8_caps():
    features = load_macro_features()
    mom_df = build_momentum_table(UNIVERSE_V8)
    etf_scores = get_all_etf_scores()
    weights, selections, cash_months = select_v8(
        mom_df, features, etf_scores, top_n_base=2,
        max_crypto_weight=0.20, max_single_asset=0.30,
    )
    CRYPTO = {"BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "XRPUSD", "ADAUSD"}
    for month in weights.index:
        row = weights.loc[month]
        active = row[row > 0]
        if len(active) > 0:
            assert active.max() <= 0.30 + 1e-9, (month, active.max())
        crypto_sum = row[row.index.isin(CRYPTO)].sum()
        assert crypto_sum <= 0.20 + 1e-9, (month, crypto_sum)

if __name__ == "__main__":
    test_select_v8_caps()
    print("OK")
