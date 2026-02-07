"""Generate comprehensive PDF reports from layer sweep results.

Designed to be readable by anyone  - CS students, ML beginners, non-technical
collaborators. Each section explains what the data means, not just what it is.
"""

from __future__ import annotations

import io
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from fpdf import FPDF

from neuronscope.experiments.schema import ExperimentResult
from neuronscope.analysis.insights import generate_sweep_insights
from neuronscope.reports.utils import safe_text


# -- Colour palette (matches frontend dark theme) --
BLUE = (59, 130, 246)
AMBER = (245, 158, 11)
DARK_BG = (24, 24, 32)
CARD_BG = (39, 39, 42)
TEXT = (228, 228, 231)
TEXT_MUTED = (161, 161, 170)
RED = (239, 68, 68)
GREEN = (34, 197, 94)


class SweepReport(FPDF):
    """Custom PDF class with NeuronScope branding."""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*TEXT_MUTED)
        self.cell(0, 8, "NeuronScope  - Layer Sweep Report", align="L")
        self.cell(0, 8, datetime.now().strftime("%Y-%m-%d %H:%M"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*TEXT_MUTED)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*TEXT_MUTED)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_sweep_pdf(results: list[ExperimentResult]) -> bytes:
    """Generate a comprehensive PDF report from sweep results.

    Returns the PDF as bytes (ready to stream as a response).
    """
    if not results:
        raise ValueError("No results to report")

    config = results[0].config
    insights = generate_sweep_insights(results)

    pdf = SweepReport(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title Section ──
    _title_section(pdf, config, results)

    # ── What Is This Report? ──
    _education_intro(pdf)

    # ── Chart ──
    _chart_section(pdf, results)

    # ── Key Findings (Insights) ──
    _insights_section(pdf, insights)

    # ── Per-Layer Table ──
    _results_table(pdf, results)

    # ── Glossary ──
    pdf.add_page()
    _glossary(pdf)

    return pdf.output()


def _title_section(pdf: SweepReport, config, results: list[ExperimentResult]):
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*TEXT)
    pdf.cell(0, 14, "Layer Sweep Analysis", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*TEXT_MUTED)

    base_input = config.base_input
    if len(base_input) > 80:
        base_input = base_input[:77] + "..."

    component = config.interventions[0].target_component.replace("_", " ").title() if config.interventions else "N/A"
    intervention = config.interventions[0].intervention_type.title() if config.interventions else "N/A"

    clean_token = results[0].clean_output_token if results else "?"
    changed_count = sum(1 for r in results if r.top_token_changed)
    peak_result = max(results, key=lambda r: r.kl_divergence)
    peak_layer = peak_result.config.interventions[0].target_layer if peak_result.config.interventions else 0

    lines = [
        f'Prompt: "{base_input}"',
        f"Clean prediction: \"{clean_token}\" (p={results[0].clean_output_prob:.1%})",
        f"Component: {component}  |  Intervention: {intervention}",
        f"Layers tested: {len(results)}  |  Prediction flipped in {changed_count}/{len(results)} layers",
        f"Peak effect: Layer {peak_layer} (KL = {peak_result.kl_divergence:.4f})",
        f"Config hash: {results[0].config_hash}",
    ]
    for line in lines:
        pdf.cell(0, 6, safe_text(line), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)


def _education_intro(pdf: SweepReport):
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 10, "What Is This Report?", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*TEXT)
    text = (
        "This report shows the results of a layer sweep  - a systematic test where "
        "we disable one component of the AI model at a time and measure how much the "
        "output changes. Each layer in the model processes text sequentially, like "
        "floors in a factory. By disabling each floor one at a time, we can figure "
        "out which floors are essential for producing the correct answer.\n\n"
        "The key metric is KL Divergence  - a number measuring how much the model's "
        "entire prediction changed. KL = 0 means nothing changed. KL > 1 means a "
        "meaningful effect. KL > 10 means a major disruption. When the bar is highlighted "
        "(amber), the model's #1 prediction actually flipped to a different word."
    )
    pdf.multi_cell(0, 5, text)
    pdf.ln(4)


def _chart_section(pdf: SweepReport, results: list[ExperimentResult]):
    """Render the KL divergence bar chart as an image and embed it."""
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 10, "KL Divergence Across Layers", new_x="LMARGIN", new_y="NEXT")

    # Build chart with matplotlib
    layers = [r.config.interventions[0].target_layer for r in results]
    kls = [r.kl_divergence for r in results]
    changed = [r.top_token_changed for r in results]

    fig, ax = plt.subplots(figsize=(7.5, 2.8), dpi=150)
    fig.patch.set_facecolor("#18181f")
    ax.set_facecolor("#18181f")

    colors = ["#f59e0b" if c else "#3b82f6" for c in changed]
    ax.bar(layers, kls, color=colors, width=0.8, edgecolor="none")

    ax.set_xlabel("Layer", color="#a1a1aa", fontsize=9)
    ax.set_ylabel("KL Divergence", color="#a1a1aa", fontsize=9)
    ax.tick_params(colors="#a1a1aa", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#3f3f46")
    ax.spines["left"].set_color("#3f3f46")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#f59e0b", label="Top token changed"),
        Patch(facecolor="#3b82f6", label="Top token preserved"),
    ]
    leg = ax.legend(handles=legend_elements, loc="upper right", fontsize=7,
                    facecolor="#27272a", edgecolor="#3f3f46", labelcolor="#a1a1aa")

    plt.tight_layout()

    # Save to temp file and embed
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        pdf.image(tmp.name, x=pdf.l_margin, w=pdf.w - pdf.l_margin - pdf.r_margin)

    pdf.ln(6)


def _insights_section(pdf: SweepReport, insights: list[dict]):
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 10, "Key Findings", new_x="LMARGIN", new_y="NEXT")

    type_colors = {
        "critical": RED,
        "notable": AMBER,
        "info": BLUE,
    }
    type_labels = {
        "critical": "IMPORTANT",
        "notable": "NOTABLE",
        "info": "INFO",
    }

    for insight in insights:
        color = type_colors.get(insight["type"], BLUE)
        label = type_labels.get(insight["type"], "INFO")

        # Check if we need a new page
        if pdf.get_y() > 250:
            pdf.add_page()

        # Badge
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*color)
        pdf.cell(20, 5, f"[{label}]", new_x="END")

        # Title
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*TEXT)
        pdf.cell(0, 5, safe_text(f"  {insight['title']}"), new_x="LMARGIN", new_y="NEXT")

        # Detail
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*TEXT_MUTED)
        pdf.multi_cell(0, 4.5, safe_text(insight["detail"]))
        pdf.ln(3)


def _results_table(pdf: SweepReport, results: list[ExperimentResult]):
    if pdf.get_y() > 200:
        pdf.add_page()

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 10, "Per-Layer Results", new_x="LMARGIN", new_y="NEXT")

    # Explanation
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*TEXT_MUTED)
    pdf.multi_cell(0, 4.5, (
        "Each row shows the effect of disabling the component at that layer. "
        "'Clean Token' is the model's normal prediction. 'Intervention Token' is "
        "what the model predicted after the component was disabled. 'Changed' means "
        "the model's top prediction flipped to a completely different word."
    ))
    pdf.ln(3)

    # Table header
    col_widths = [18, 28, 22, 20, 30, 30, 22]
    headers = ["Layer", "KL Div", "Changed", "Effect", "Clean Token", "Interv. Token", "Duration"]

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*TEXT)
    pdf.set_fill_color(39, 39, 42)

    for i, (header, width) in enumerate(zip(headers, col_widths)):
        pdf.cell(width, 6, header, border=0, fill=True, align="C",
                 new_x="END" if i < len(headers) - 1 else "LMARGIN",
                 new_y="TOP" if i < len(headers) - 1 else "NEXT")

    # Table rows
    pdf.set_font("Helvetica", "", 8)
    for j, r in enumerate(results):
        if pdf.get_y() > 270:
            pdf.add_page()
            # Re-print header
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(39, 39, 42)
            for i, (header, width) in enumerate(zip(headers, col_widths)):
                pdf.cell(width, 6, header, border=0, fill=True, align="C",
                         new_x="END" if i < len(headers) - 1 else "LMARGIN",
                         new_y="TOP" if i < len(headers) - 1 else "NEXT")
            pdf.set_font("Helvetica", "", 8)

        layer = r.config.interventions[0].target_layer if r.config.interventions else 0
        changed_str = "YES" if r.top_token_changed else "no"
        kl = f"{r.kl_divergence:.4f}"
        clean_tok = r.clean_output_token.strip()[:8]
        int_tok = r.intervention_output_token.strip()[:8]
        duration = f"{r.duration_seconds:.1f}s"

        # Effect category
        if r.kl_divergence > 10:
            effect = "Critical"
            eff_color = RED
        elif r.kl_divergence > 5:
            effect = "High"
            eff_color = AMBER
        elif r.kl_divergence > 1:
            effect = "Moderate"
            eff_color = TEXT
        else:
            effect = "Low"
            eff_color = TEXT_MUTED

        # Alternating row shade
        if j % 2 == 0:
            pdf.set_fill_color(30, 30, 36)
        else:
            pdf.set_fill_color(24, 24, 32)

        row_data = [str(layer), kl, changed_str, effect, f'"{safe_text(clean_tok)}"', f'"{safe_text(int_tok)}"', duration]
        row_colors = [TEXT, TEXT, RED if r.top_token_changed else TEXT_MUTED, eff_color, TEXT, TEXT, TEXT_MUTED]

        for i, (val, width, color) in enumerate(zip(row_data, col_widths, row_colors)):
            pdf.set_text_color(*color)
            pdf.cell(width, 5.5, val, border=0, fill=True, align="C",
                     new_x="END" if i < len(row_data) - 1 else "LMARGIN",
                     new_y="TOP" if i < len(row_data) - 1 else "NEXT")

    pdf.ln(4)


def _glossary(pdf: SweepReport):
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 12, "Glossary", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*TEXT_MUTED)
    pdf.multi_cell(0, 4.5, (
        "This glossary explains the key terms used in this report, "
        "written for people who may not have a machine learning background."
    ))
    pdf.ln(4)

    terms = [
        ("Layer", (
            "One processing step in the model's pipeline. Text is processed "
            "through all layers sequentially  - like floors in a factory where "
            "each floor adds more refinement."
        )),
        ("MLP (Multi-Layer Perceptron)", (
            "The 'knowledge storage' component in each layer. Research shows "
            "that factual knowledge (like 'Eiffel Tower -> Paris') is often "
            "stored in MLP layers."
        )),
        ("Attention", (
            "The component that decides which words in the input to focus on. "
            "When you read 'The cat sat on the ___', attention helps the model "
            "look back at 'cat' and 'sat' to predict 'mat'."
        )),
        ("Zero Ablation", (
            "Completely removing a component's output by setting it to zero. "
            "Like unplugging one wire in a circuit  - if the lights go out, "
            "that wire was important."
        )),
        ("KL Divergence", (
            "A number measuring how different two probability distributions are. "
            "KL = 0 means the intervention had no effect. KL > 1 is meaningful. "
            "KL > 10 is a major disruption. Higher = the component matters more."
        )),
        ("Top Token", (
            "The word the model considers most likely to come next. When "
            "the 'top token changed', the model's #1 prediction flipped to "
            "a completely different word."
        )),
        ("Logit", (
            "The raw score the model assigns to each possible next word. "
            "Higher logit = the model thinks that word is more likely. "
            "Logits are converted to probabilities using the softmax function."
        )),
        ("Probability", (
            "The model's confidence that a particular word is the right "
            "next word, expressed as a percentage (0-100%). A probability "
            "of 95% means the model is very confident."
        )),
        ("Config Hash", (
            "A unique fingerprint of your experiment setup. If someone runs "
            "the same experiment and gets the same hash, the results should "
            "be identical. This guarantees reproducibility."
        )),
        ("Activation Patching", (
            "A technique where you run the model on two different inputs, "
            "then swap the internal values from one into the other at a "
            "specific point. If the output changes, that point carries the "
            "information that distinguishes the two inputs."
        )),
        ("Residual Stream", (
            "The main 'highway' of information flowing through the model. "
            "Each layer reads from and writes to this stream. It's called "
            "'residual' because each layer adds its contribution on top of "
            "what came before."
        )),
    ]

    for term, definition in terms:
        if pdf.get_y() > 260:
            pdf.add_page()

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*TEXT)
        pdf.cell(0, 6, term, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*TEXT_MUTED)
        pdf.multi_cell(0, 4.5, definition)
        pdf.ln(2)

    # Footer
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*TEXT_MUTED)
    pdf.multi_cell(0, 4, (
        "Generated by NeuronScope  - an open-source mechanistic interpretability tool "
        "for causal intervention on LLM internals. "
        "Understanding is measured by controllability."
    ))
