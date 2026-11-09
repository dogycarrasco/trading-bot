"""Reporte PDF v9 - Bot con mejoras anti-drawdown."""
import os, json
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, KeepTogether, HRFlowable
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
IMAGES = os.path.join(RESULTS, "images")
PDF_OUT = os.path.join(ROOT, "docs", "Reporte_Bot_v9_Anti_Drawdown.pdf")
os.makedirs(os.path.dirname(PDF_OUT), exist_ok=True)

FONT_REG = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
NAVY = colors.HexColor("#1B2A4E")
BLUE = colors.HexColor("#2E86AB")
GREEN = colors.HexColor("#2A9D8F")
RED = colors.HexColor("#E63946")
ORANGE = colors.HexColor("#F4A261")
GRAY = colors.HexColor("#6C757D")
LIGHT = colors.HexColor("#F8F9FA")
DARK = colors.HexColor("#212529")

styles = getSampleStyleSheet()
TITLE = ParagraphStyle("Title", parent=styles["Title"], fontName=FONT_BOLD, fontSize=22,
                       textColor=NAVY, alignment=TA_LEFT, spaceAfter=6)
SUBTITLE = ParagraphStyle("Subtitle", parent=styles["Normal"], fontName=FONT_REG, fontSize=11,
                          textColor=GRAY, alignment=TA_LEFT, spaceAfter=14)
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName=FONT_BOLD, fontSize=15,
                    textColor=NAVY, spaceBefore=14, spaceAfter=8, alignment=TA_LEFT)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=FONT_BOLD, fontSize=12,
                    textColor=BLUE, spaceBefore=10, spaceAfter=4, alignment=TA_LEFT)
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontName=FONT_REG, fontSize=10,
                      textColor=DARK, alignment=TA_JUSTIFY, leading=14, spaceAfter=6)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.5, textColor=GRAY, alignment=TA_LEFT)
CAPTION = ParagraphStyle("Caption", parent=BODY, fontSize=9, textColor=GRAY, alignment=TA_CENTER,
                         spaceAfter=10)
KPI_LABEL = ParagraphStyle("KpiLabel", parent=BODY, fontSize=8.5, textColor=GRAY,
                           alignment=TA_CENTER, spaceAfter=0)
KPI_VAL = ParagraphStyle("KpiVal", parent=BODY, fontName=FONT_BOLD, fontSize=14,
                         textColor=NAVY, alignment=TA_CENTER, spaceAfter=0)
WIN_ALERT = ParagraphStyle("WinAlert", parent=BODY, fontSize=12, textColor=GREEN, alignment=TA_CENTER,
                           spaceAfter=10, spaceBefore=6)


def load_summary():
    with open(os.path.join(RESULTS, "v9_summary.json")) as f:
        return json.load(f)


def kpi_box(label, value, color=NAVY, w=3.3):
    val_style = ParagraphStyle("v", parent=KPI_VAL, textColor=color, fontSize=13)
    t = Table([[Paragraph(label, KPI_LABEL)], [Paragraph(value, val_style)]],
              colWidths=[w * cm], rowHeights=[0.5 * cm, 0.85 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DEE2E6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_REG, 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(2 * cm, A4[1] - 1 * cm, "Bot v9 - Anti-drawdown")
    canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1 * cm, datetime.now().strftime("%Y-%m-%d"))
    canvas.setStrokeColor(colors.HexColor("#DEE2E6"))
    canvas.line(2 * cm, A4[1] - 1.2 * cm, A4[0] - 2 * cm, A4[1] - 1.2 * cm)
    canvas.drawString(2 * cm, 1 * cm, "Benchmark: SPY buy&hold + inflacion real EC 17% anual")
    canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, f"Pagina {doc.page}")
    canvas.line(2 * cm, 1.3 * cm, A4[0] - 2 * cm, 1.3 * cm)
    canvas.restoreState()


def build_pdf():
    summary = load_summary()
    bm = summary["bot_metrics"]
    sm = summary["spy_metrics"]
    bot_y = summary["bot_annual_returns"]
    spy_y = summary["spy_annual_returns"]
    stats = summary.get("stats", {})

    doc = SimpleDocTemplate(
        PDF_OUT, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Bot v9 - Anti Drawdown",
    )

    flow = []

    flow.append(Spacer(1, 0.3 * cm))
    flow.append(Paragraph("Bot v9 - Anti-Drawdown", TITLE))
    flow.append(Paragraph("Stop-loss mensual + crypto winter filter + smart money exit + trend filter EMA200", SUBTITLE))
    flow.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=10))

    bot_wins_spy = sum(1 for y, v in bot_y.items() if v > spy_y.get(y, 0))
    bot_wins_infl = sum(1 for y, v in bot_y.items() if v > 17.0)
    delta_spy = bm["cagr_pct"] - sm["cagr_pct"]
    delta_infl = bm["cagr_pct"] - 17.0

    flow.append(Paragraph(
        f"<b>RESULTADO 20 AnOS (v9):</b><br/>"
        f"&bull; Bot v9: ${bm['final_equity']:.0f} ({bm['total_return_pct']:+.1f}%, CAGR <b>{bm['cagr_pct']:+.2f}%</b>)<br/>"
        f"&bull; SPY: ${sm['final_equity']:.0f} ({sm['total_return_pct']:+.1f}%, CAGR {sm['cagr_pct']:+.2f}%)<br/>"
        f"&bull; Bot le gana al SPY en <b>{bot_wins_spy}/{len(bot_y)} anos</b><br/>"
        f"&bull; Bot supera inflacion EC 17% en {bot_wins_infl}/{len(bot_y)} anos<br/>"
        f"&bull; <b>PRIMERA VEZ que el bot supera el 17% de inflacion real EC</b> (+{delta_infl:.2f}pp)",
        WIN_ALERT
    ))
    flow.append(Spacer(1, 6))

    flow.append(Paragraph("KPIs principales", H2))
    kpi_row = Table([[
        kpi_box("Capital inicial", "$500", w=3.2),
        kpi_box("Capital final bot", f"${bm['final_equity']:.0f}", color=GREEN, w=3.2),
        kpi_box("Capital final SPY", f"${sm['final_equity']:.0f}", color=ORANGE, w=3.2),
        kpi_box("CAGR bot", f"{bm['cagr_pct']:+.2f}%", color=GREEN, w=3.2),
        kpi_box("CAGR SPY", f"{sm['cagr_pct']:+.2f}%", w=3.2),
    ]], colWidths=[3.4 * cm] * 5)
    kpi_row.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    flow.append(kpi_row)
    flow.append(Spacer(1, 6))
    kpi_row2 = Table([[
        kpi_box("Sharpe bot", f"{bm['sharpe']:.2f}", w=3.2),
        kpi_box("Sharpe SPY", f"{sm['sharpe']:.2f}", w=3.2),
        kpi_box("Max DD bot", f"{bm['max_drawdown_pct']:.1f}%", color=GREEN, w=3.2),
        kpi_box("Max DD SPY", f"{sm['max_drawdown_pct']:.1f}%", color=RED, w=3.2),
        kpi_box("Bot vs Infl 17%", f"{delta_infl:+.2f}pp", color=GREEN if delta_infl > 0 else RED, w=3.2),
    ]], colWidths=[3.4 * cm] * 5)
    kpi_row2.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    flow.append(kpi_row2)
    flow.append(Spacer(1, 12))

    if os.path.exists(os.path.join(IMAGES, "v9_equity.png")):
        flow.append(Image(os.path.join(IMAGES, "v9_equity.png"), width=17 * cm, height=7.8 * cm))
        flow.append(Paragraph("Figura 1. Equity curves 20 anos: bot v9 (azul) vs SPY (rojo punteado).", CAPTION))

    flow.append(PageBreak())

    flow.append(Paragraph("1. Resumen ejecutivo", H1))
    flow.append(Paragraph(
        f"v9 aplica 4 mejoras anti-drawdown: stop-loss mensual (-15%), crypto winter filter (BTC<EMA200m), "
        f"smart money exit (SPY<-5% mensual), y trend filter EMA200. "
        f"CAGR {bm['cagr_pct']:+.2f}% vs SPY {sm['cagr_pct']:+.2f}% ({delta_spy:+.2f}pp). "
        f"<b>Primera version que supera el 17% de inflacion EC</b> ({delta_infl:+.2f}pp).",
        BODY))
    flow.append(Paragraph(
        f"<b>Stats:</b><br/>"
        f"&bull; Stop-loss events: {stats.get('stop_loss_events', 0)}<br/>"
        f"&bull; Meses en cash: {stats.get('cash_months', 0)}",
        BODY))

    if os.path.exists(os.path.join(IMAGES, "v9_evolution.png")):
        flow.append(PageBreak())
        flow.append(Paragraph("2. Evolucion v5 -> v9", H1))
        flow.append(Image(os.path.join(IMAGES, "v9_evolution.png"), width=17 * cm, height=7.5 * cm))
        flow.append(Paragraph("Figura 2. v9 es la primera version que supera el 17% de inflacion EC.", CAPTION))

    if os.path.exists(os.path.join(IMAGES, "v9_yoy_table.png")):
        flow.append(PageBreak())
        flow.append(Paragraph("3. Ano por ano (2007-2026)", H1))
        flow.append(Image(os.path.join(IMAGES, "v9_yoy_table.png"), width=17 * cm, height=14 * cm))

    if os.path.exists(os.path.join(IMAGES, "v9_yoy.png")):
        flow.append(PageBreak())
        flow.append(Paragraph("4. Retorno anual", H1))
        flow.append(Image(os.path.join(IMAGES, "v9_yoy.png"), width=17 * cm, height=8 * cm))

    flow.append(PageBreak())
    flow.append(Paragraph("5. Conclusion", H1))
    flow.append(Paragraph(
        f"v9 cumple el objetivo: CAGR {bm['cagr_pct']:+.2f}% supera el 17% de inflacion real EC por {delta_infl:+.2f}pp. "
        f"En 20 anos, $500 ? ${bm['final_equity']:.0f} vs ${sm['final_equity']:.0f} del SPY. "
        f"Diferencia +${bm['final_equity']-sm['final_equity']:.0f}.",
        BODY))
    flow.append(Paragraph(
        "<b>Siguientes pasos:</b><br/>"
        "&bull; Activar paper trading v9 (bot/main.py)<br/>"
        "&bull; Validar 30 dias en paper antes de cualquier capital real",
        BODY))

    flow.append(Spacer(1, 10))
    flow.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
    flow.append(Spacer(1, 4))
    flow.append(Paragraph(
        "Disclaimer: backtest sobre datos historicos. El rendimiento pasado no garantiza resultados futuros. "
        "Inflacion real EC 17% anual es la tasa bancaria de Ecuador. "
        "Cripto y futuros implican riesgo de perdida total.",
        SMALL))

    doc.build(flow, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"PDF generado: {PDF_OUT}")
    print(f"Tamano: {os.path.getsize(PDF_OUT) / 1024:.1f} KB")


if __name__ == "__main__":
    build_pdf()