"""Generate a PDF report from a single experiment result."""

from __future__ import annotations

from datetime import datetime

from fpdf import FPDF

from neuronscope.experiments.schema import ExperimentResult
from neuronscope.analysis.insights import generate_insights


# Colours (same as sweep_report)
BLUE = (59, 130, 246)
AMBER = (245, 158, 11)
TEXT = (228, 228, 231)
TEXT_MUTED = (161, 161, 170)
RED = (239, 68, 68)
GREEN = (34, 197, 94)


class ExperimentReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*TEXT_MUTED)
        self.cell(0, 8, "NeuronScope  - Experiment Report", align="L")
        self.cell(0, 8, datetime.now().strftime("%Y-%m-%d %H:%M"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*TEXT_MUTED)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*TEXT_MUTED)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_experiment_pdf(result: ExperimentResult) -> bytes:
    """Generate a single-experiment PDF report."""
    insights = generate_insights(result)

    pdf = ExperimentReport(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    config = result.config
    spec = config.interventions[0] if config.interventions else None

    # ── Title ──
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*TEXT)
    pdf.cell(0, 12, "Experiment Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*TEXT_MUTED)

    base_input = config.base_input
    if len(base_input) > 80:
        base_input = base_input[:77] + "..."

    lines = [
        f'Prompt: "{base_input}"',
        f"Experiment: {config.name or result.id}",
    ]
    if spec:
        component = spec.target_component.replace("_", " ").title()
        lines.append(f"Intervention: {spec.intervention_type.title()} on Layer {spec.target_layer} {component}")
        if spec.target_position is not None:
            lines.append(f"Token Position: {spec.target_position}")

    lines += [
        f"Seed: {config.seed}  |  Duration: {result.duration_seconds:.2f}s  |  Device: {result.device}",
        f"Config hash: {result.config_hash}",
    ]

    for line in lines:
        pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── Key Metrics ──
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 10, "Key Metrics", new_x="LMARGIN", new_y="NEXT")

    metrics = [
        ("KL Divergence", f"{result.kl_divergence:.4f}", _kl_label(result.kl_divergence)),
        ("Top Token Changed", "YES" if result.top_token_changed else "NO", ""),
        ("Clean Output", f'"{result.clean_output_token}" (p={result.clean_output_prob:.4f})', ""),
        ("Intervention Output", f'"{result.intervention_output_token}" (p={result.intervention_output_prob:.4f})', ""),
    ]

    for label, value, note in metrics:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*TEXT)
        pdf.cell(50, 6, label + ":", new_x="END")
        pdf.set_font("Helvetica", "", 10)
        color = AMBER if ("YES" in value or result.kl_divergence > 1) and "Changed" in label else TEXT
        pdf.set_text_color(*color)
        pdf.cell(60, 6, value, new_x="END")
        if note:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*TEXT_MUTED)
            pdf.cell(0, 6, note, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.cell(0, 6, "", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)

    # ── Top-K Comparison ──
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 10, "Top-10 Predictions: Clean vs Intervention", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*TEXT_MUTED)
    pdf.multi_cell(0, 4.5, (
        "These tables show the model's top 10 most-likely next words, "
        "before and after the intervention. Compare them side by side "
        "to see which words gained or lost probability."
    ))
    pdf.ln(3)

    # Side by side tables
    half_w = (pdf.w - pdf.l_margin - pdf.r_margin - 6) / 2

    y_start = pdf.get_y()

    # Clean run table
    _topk_table(pdf, "Clean Run", result.clean_top_k, pdf.l_margin, y_start, half_w)

    # Intervention run table
    _topk_table(pdf, "Intervention Run", result.intervention_top_k, pdf.l_margin + half_w + 6, y_start, half_w)

    # Move Y past both tables
    pdf.set_y(y_start + 6 + 5.5 * min(len(result.clean_top_k), 10) + 8)
    pdf.ln(4)

    # ── Rank Changes ──
    if result.rank_changes:
        if pdf.get_y() > 210:
            pdf.add_page()

        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*BLUE)
        pdf.cell(0, 10, "Significant Rank Changes", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*TEXT_MUTED)
        pdf.multi_cell(0, 4.5, (
            "Shows words that moved significantly in the model's ranking. "
            "A positive delta means the word became less likely (dropped in rank). "
            "A negative delta means it became more likely (rose in rank)."
        ))
        pdf.ln(3)

        sorted_rc = sorted(
            result.rank_changes.items(),
            key=lambda x: abs(x[1]["rank_delta"]),
            reverse=True,
        )[:20]

        col_w = [40, 35, 40, 35]
        headers = ["Token", "Clean Rank", "Intervention Rank", "Delta"]

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(39, 39, 42)
        pdf.set_text_color(*TEXT)
        for i, (h, w) in enumerate(zip(headers, col_w)):
            pdf.cell(w, 6, h, fill=True, align="C",
                     new_x="END" if i < len(headers) - 1 else "LMARGIN",
                     new_y="TOP" if i < len(headers) - 1 else "NEXT")

        pdf.set_font("Helvetica", "", 8)
        for j, (token, rc) in enumerate(sorted_rc):
            if pdf.get_y() > 270:
                pdf.add_page()

            token_display = token.strip()[:12]
            delta = rc["rank_delta"]
            delta_str = f"+{delta}" if delta > 0 else str(delta)
            delta_color = RED if delta > 0 else GREEN if delta < 0 else TEXT_MUTED

            if j % 2 == 0:
                pdf.set_fill_color(30, 30, 36)
            else:
                pdf.set_fill_color(24, 24, 32)

            row = [f'"{token_display}"', str(rc["clean_rank"]), str(rc["intervention_rank"]), delta_str]
            colors = [TEXT, TEXT_MUTED, TEXT_MUTED, delta_color]

            for i, (val, w, c) in enumerate(zip(row, col_w, colors)):
                pdf.set_text_color(*c)
                pdf.cell(w, 5.5, val, fill=True, align="C",
                         new_x="END" if i < len(row) - 1 else "LMARGIN",
                         new_y="TOP" if i < len(row) - 1 else "NEXT")

        pdf.ln(4)

    # ── Insights ──
    if pdf.get_y() > 230:
        pdf.add_page()

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 10, "What This Means", new_x="LMARGIN", new_y="NEXT")

    type_colors = {"critical": RED, "notable": AMBER, "info": BLUE}
    type_labels = {"critical": "IMPORTANT", "notable": "NOTABLE", "info": "INFO"}

    for insight in insights:
        if pdf.get_y() > 255:
            pdf.add_page()

        color = type_colors.get(insight["type"], BLUE)
        label = type_labels.get(insight["type"], "INFO")

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*color)
        pdf.cell(20, 5, f"[{label}]", new_x="END")

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*TEXT)
        pdf.cell(0, 5, f"  {insight['title']}", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*TEXT_MUTED)
        pdf.multi_cell(0, 4.5, insight["detail"])
        pdf.ln(2)

    # Footer
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*TEXT_MUTED)
    pdf.multi_cell(0, 4, (
        "Generated by NeuronScope  - an open-source mechanistic interpretability tool. "
        "Understanding is measured by controllability."
    ))

    return pdf.output()


def _topk_table(pdf, title, predictions, x, y, width):
    """Render a top-k table at a specific position."""
    col_widths = [width * 0.12, width * 0.35, width * 0.27, width * 0.26]

    pdf.set_xy(x, y)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*TEXT)
    pdf.cell(width, 6, title, new_x="LMARGIN", new_y="NEXT")

    pdf.set_xy(x, y + 6)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_fill_color(39, 39, 42)
    headers = ["#", "Token", "Logit", "Prob"]
    for i, (h, w) in enumerate(zip(headers, col_widths)):
        pdf.cell(w, 5, h, fill=True, align="C", new_x="END", new_y="TOP")
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    for j, p in enumerate(predictions[:10]):
        pdf.set_xy(x, y + 11 + j * 5.5)
        if j % 2 == 0:
            pdf.set_fill_color(30, 30, 36)
        else:
            pdf.set_fill_color(24, 24, 32)

        token_str = p.token.strip()[:10]
        row = [str(j + 1), f'"{token_str}"', f"{p.logit:.2f}", f"{p.prob * 100:.1f}%"]

        for i, (val, w) in enumerate(zip(row, col_widths)):
            pdf.set_text_color(*TEXT)
            pdf.cell(w, 5.5, val, fill=True, align="C", new_x="END", new_y="TOP")
        pdf.ln()


def _kl_label(kl: float) -> str:
    if kl > 10:
        return "(Critical effect)"
    elif kl > 5:
        return "(High effect)"
    elif kl > 1:
        return "(Moderate effect)"
    else:
        return "(Low effect)"
