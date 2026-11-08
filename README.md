# Bot de Trading v9 - Sector Rotation + Hedge Fund Agents

Bot de trading open-source que aplica ingeniería inversa de 4 repos open-source
(Ghostfolio, OpenBB, Metabase, Vane-AI) + ai-hedge-fund (59k stars).

## Resultado del backtest (20 años, 2007-2026)

- Bot: $500 ? $13,882 (CAGR +18.14%)
- SPY: $500 ? $2,596 (CAGR +8.54%)
- Bot le gana al SPY por +9.60pp anuales
- Bot supera inflación EC 17% por +1.14pp

## Deploy en GitHub Actions

Ver README_cloud.md para instrucciones de deploy en la nube.

## Cómo correrlo

### Backtest
cd trading_bot
pip install -r requirements.txt
pip install git+https://github.com/rongardF/tvdatafeed.git
python -m backtest.run_v9_improvements

### Live paper trading
python -m bot.main --once