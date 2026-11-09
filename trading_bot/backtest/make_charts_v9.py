"""Graficas v9 - Bot con mejoras anti-drawdown."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
IMAGES = os.path.join(RESULTS, "images")
os.makedirs(IMAGES, exist_ok=True)


def load_summary():
    with open(os.path.join(RESULTS, "v9_summary.json")) as f:
        return json.load(f)


def plot_yoy(summary):
    bot_y = summary["bot_annual_returns"]
    spy_y = summary["spy_annual_returns"]
    years = sorted(set(bot_y.keys()) | set(spy_y.keys()))
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(years))
    width = 0.28
    bot_vals = [bot_y.get(y, 0) for y in years]
    spy_vals = [spy_y.get(y, 0) for y in years]
    infl_vals = [17.0] * len(years)
    bars1 = ax.bar(x - width, bot_vals, width, label='Bot v9', color='#2E86AB', edgecolor='white')
    bars2 = ax.bar(x, spy_vals, width, label='SPY buy&hold', color='#E63946', edgecolor='white')
    bars3 = ax.bar(x + width, infl_vals, width, label='Inflacion EC 17%', color='#F4A261', edgecolor='white')
    for bars in [bars1, bars2]:
        for b in bars:
            h = b.get_height()
            if abs(h) > 0.5:
                ax.annotate(f'{h:+.0f}%' if abs(h) > 5 else f'{h:+.1f}%',
                            xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 3 if h >= 0 else -12),
                            textcoords="offset points", ha='center', fontsize=7, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.7)
    ax.axhline(y=17, color='#F4A261', linewidth=0.8, linestyle=':', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, fontsize=9)
    ax.set_ylabel('Retorno anual (%)')
    ax.set_title(f'Bot v9 vs SPY vs Inflacion EC (2007-2026)\n'
                 f'CAGR {summary["bot_metrics"]["cagr_pct"]:+.2f}% vs SPY {summary["spy_metrics"]["cagr_pct"]:+.2f}%',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES, "v9_yoy.png"), dpi=110)
    plt.close()
    print("  v9_yoy.png")


def plot_equity(summary):
    bot_eq = pd.read_csv(os.path.join(RESULTS, "v9_portfolio_equity.csv"), index_col=0, parse_dates=True)
    spy_eq = pd.read_csv(os.path.join(RESULTS, "v9_spy_buyhold.csv"), index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(13, 6))
    bot_col = bot_eq.columns[0]
    spy_col = spy_eq.columns[0]
    ax.plot(bot_eq.index, bot_eq[bot_col], color='#2E86AB', linewidth=2.4,
            label=f'Bot v9 (final ${float(bot_eq[bot_col].iloc[-1]):.0f})')
    ax.plot(spy_eq.index, spy_eq[spy_col], color='#E63946', linewidth=2.0, linestyle='--',
            label=f'SPY (final ${float(spy_eq[spy_col].iloc[-1]):.0f})')
    ax.axhline(y=500, color='gray', linestyle=':', linewidth=0.8, alpha=0.7, label='Capital inicial ($500)')
    final_bot = float(bot_eq[bot_col].iloc[-1])
    final_spy = float(spy_eq[spy_col].iloc[-1])
    ax.set_title(f'$500 inicial -> Bot: ${final_bot:.0f} (+{(final_bot/500-1)*100:.0f}%)  |  SPY: ${final_spy:.0f} (+{(final_spy/500-1)*100:.0f}%)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Capital (USD)')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    common = bot_eq.index.intersection(spy_eq.index)
    bot_c = bot_eq[bot_col].reindex(common)
    spy_c = spy_eq[spy_col].reindex(common).ffill()
    ax.fill_between(common, spy_c, bot_c, where=bot_c > spy_c,
                    color='#2A9D8F', alpha=0.2, interpolate=True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES, "v9_equity.png"), dpi=110)
    plt.close()
    print("  v9_equity.png")


def plot_yoy_table(summary):
    bot_y = summary["bot_annual_returns"]
    spy_y = summary["spy_annual_returns"]
    years = sorted(set(bot_y.keys()) | set(spy_y.keys()))
    fig, ax = plt.subplots(figsize=(12, 0.32 * len(years) + 1.5))
    ax.axis("off")
    rows = []
    for y in years:
        bot = bot_y.get(y, 0)
        spy = spy_y.get(y, 0)
        infl = 17.0
        diff_spy = bot - spy
        diff_infl = bot - infl
        win_spy = "GANA" if diff_spy > 0 else "PIERDE"
        win_infl = "GANA" if diff_infl > 0 else "PIERDE"
        rows.append([y, f"{bot:+.1f}%", f"{spy:+.1f}%", f"{infl:.0f}%",
                     f"{diff_spy:+.1f}pp", win_spy, win_infl])
    cols = ["Ano", "Bot v9", "SPY", "Inflacion\nEC", "vs SPY", "vs SPY?", "vs Infl?"]
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for i, row in enumerate(rows):
        win_spy = row[5]
        win_infl = row[6]
        table[(i + 1, 5)].set_facecolor("#D4EDDA" if win_spy == "GANA" else "#F8D7DA")
        table[(i + 1, 6)].set_facecolor("#D4EDDA" if win_infl == "GANA" else "#F8D7DA")
        bot_val = float(row[1].replace('%', '').replace('+', ''))
        spy_val = float(row[2].replace('%', '').replace('+', ''))
        table[(i + 1, 1)].set_facecolor("#D4EDDA" if bot_val > 0 else "#F8D7DA")
        table[(i + 1, 2)].set_facecolor("#D4EDDA" if spy_val > 0 else "#F8D7DA")
    for j in range(len(cols)):
        table[(0, j)].set_facecolor("#1B2A4E")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    ax.set_title("20 anos (2007-2026): Bot v9 vs SPY vs Inflacion real EC (17%)",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES, "v9_yoy_table.png"), dpi=110, bbox_inches="tight")
    plt.close()
    print("  v9_yoy_table.png")


def plot_evolution():
    fig, ax = plt.subplots(figsize=(11, 5))
    versions = ['v5\n(TV)', 'v6\n(VIX)', 'v7\n(macro)', 'v8\n(agentes)', 'v9\n(anti-DD)']
    cagr = [8.90, 10.93, 11.00, 13.42, 18.14]
    max_dd = [-24.55, -40.92, -40.0, -50.0, -36.62]
    infl = [17, 17, 17, 17, 17]
    x = np.arange(len(versions))
    width = 0.28
    bars1 = ax.bar(x - width, cagr, width, label='CAGR bot', color='#2E86AB', edgecolor='white')
    bars2 = ax.bar(x, max_dd, width, label='Max DD bot', color='#E63946', edgecolor='white')
    bars3 = ax.bar(x + width, infl, width, label='Inflacion EC 17%', color='#F4A261', edgecolor='white')
    for bars in [bars1, bars2, bars3]:
        for b in bars:
            h = b.get_height()
            ax.annotate(f'{h:+.1f}%', xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                        ha='center', fontsize=8, fontweight='bold')
    ax.axhline(y=17, color='#F4A261', linewidth=0.8, linestyle=':', alpha=0.6)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=9)
    ax.set_ylabel('%')
    ax.set_title('Evolucion del bot v5 -> v9: v9 es la PRIMERA version que supera el 17% EC',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES, "v9_evolution.png"), dpi=110)
    plt.close()
    print("  v9_evolution.png")


def main():
    print("Generando graficas v9...")
    summary = load_summary()
    plot_yoy(summary)
    plot_equity(summary)
    plot_yoy_table(summary)
    plot_evolution()
    print(f"Listo.")


if __name__ == "__main__":
    main()