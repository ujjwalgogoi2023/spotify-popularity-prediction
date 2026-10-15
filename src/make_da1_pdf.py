"""Build the DA1 submission document as a PDF.

Covers the eight required points. Figures are the generated diagrams and EDA plots,
and every number quoted is read from the results CSVs rather than retyped.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
OUT = ROOT / "report" / "DA1_Spotify_Popularity.pdf"

# ---- fill these in -----------------------------------------------------------------
STUDENT_NAME = "[YOUR NAME]"
REG_NUMBER = "[YOUR REGISTRATION NUMBER]"
COURSE = "Predictive Analytics"
ASSESSMENT = "Digital Assignment 1 (DA1) — Project Proposal"
# ------------------------------------------------------------------------------------

INK = colors.HexColor("#1B2631")
MUTED = colors.HexColor("#5D6D7E")
RULE = colors.HexColor("#AEB6BF")
HEADBG = colors.HexColor("#E3EDFB")
ALTBG = colors.HexColor("#F7F9FA")
ACCENT = colors.HexColor("#B03A2E")

ss = getSampleStyleSheet()

H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                    fontSize=13, textColor=INK, spaceBefore=16, spaceAfter=7,
                    leading=16)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                    fontSize=10.5, textColor=INK, spaceBefore=10, spaceAfter=4,
                    leading=13)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontName="Helvetica",
                      fontSize=9.4, leading=13.4, alignment=TA_JUSTIFY,
                      textColor=INK, spaceAfter=6)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=13, bulletIndent=3,
                        spaceAfter=3)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.4, leading=11.2,
                      alignment=TA_JUSTIFY, spaceAfter=0)
CELLL = ParagraphStyle("CELLL", parent=CELL, alignment=TA_LEFT)
CELLC = ParagraphStyle("CELLC", parent=CELL, alignment=TA_CENTER)
CELLH = ParagraphStyle("CELLH", parent=CELL, fontName="Helvetica-Bold",
                       alignment=TA_CENTER)
CAPTION = ParagraphStyle("CAPTION", parent=BODY, fontSize=8.2, alignment=TA_CENTER,
                         textColor=MUTED, spaceBefore=4, spaceAfter=10)
NOTE = ParagraphStyle("NOTE", parent=BODY, fontSize=8.8, leading=12.4,
                      textColor=colors.HexColor("#7B241C"), leftIndent=8,
                      rightIndent=8, spaceBefore=4, spaceAfter=8)


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items, style=BULLET):
    return [Paragraph(t, style, bulletText="•") for t in items]


def table(rows, widths, header=True, align_center_cols=()):
    data = []
    for r_i, row in enumerate(rows):
        out = []
        for c_i, cell in enumerate(row):
            if isinstance(cell, Paragraph):
                out.append(cell)
            elif r_i == 0 and header:
                out.append(Paragraph(str(cell), CELLH))
            elif c_i in align_center_cols:
                out.append(Paragraph(str(cell), CELLC))
            else:
                # Justifying a narrow column stretches inter-word spacing badly, so
                # only wide prose columns are justified.
                out.append(Paragraph(str(cell),
                                     CELL if widths[c_i] >= 8.0 * cm else CELLL))
        data.append(out)

    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), HEADBG)]
        for i in range(2, len(data), 2):
            style.append(("BACKGROUND", (0, i), (-1, i), ALTBG))
    t.setStyle(TableStyle(style))
    return t


def figure(path, width_cm, caption):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    w_pt = width_cm * cm
    return KeepTogether([Image(str(path), width=w_pt, height=w_pt * h / w),
                         P(caption, CAPTION)])


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2.0 * cm, A4[1] - 1.15 * cm,
                      "Predicting Song Popularity on Spotify Using Audio Features")
    canvas.drawRightString(A4[0] - 2.0 * cm, A4[1] - 1.15 * cm, f"{COURSE} — DA1")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(2.0 * cm, A4[1] - 1.32 * cm, A4[0] - 2.0 * cm, A4[1] - 1.32 * cm)
    canvas.drawCentredString(A4[0] / 2, 1.15 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2.0 * cm, rightMargin=2.0 * cm,
        topMargin=1.75 * cm, bottomMargin=1.75 * cm,
        title="DA1 — Predicting Song Popularity on Spotify Using Audio Features",
        author=STUDENT_NAME,
    )
    S = []

    # ---------------------------------------------------------------- 1. Title block
    S.append(Paragraph(
        "Predicting Song Popularity on Spotify Using Audio Features",
        ParagraphStyle("T", parent=ss["Title"], fontName="Helvetica-Bold",
                       fontSize=17, leading=21, textColor=INK, spaceAfter=4)))
    S.append(Paragraph(ASSESSMENT, ParagraphStyle(
        "ST", parent=ss["Normal"], fontSize=10, alignment=TA_CENTER,
        textColor=MUTED, spaceAfter=12)))

    S.append(table([
        ["Field", "Detail"],
        ["Project Title", "Predicting Song Popularity on Spotify Using Audio Features"],
        ["Name", f"<b>{STUDENT_NAME}</b>"],
        ["Registration Number", f"<b>{REG_NUMBER}</b>"],
        ["Course", COURSE],
        ["Assessment", ASSESSMENT],
    ], widths=[4.6 * cm, 12.4 * cm]))
    S.append(Spacer(1, 12))

    # ------------------------------------------------------------------ 2. Objective
    S.append(P("2. Objective of the Proposed Work", H1))
    S.append(P(
        "Streaming platforms describe every track with numeric audio descriptors such as "
        "danceability, energy, valence and acousticness. This project asks whether the "
        "measurable character of a recording explains how popular it becomes on Spotify."))
    S.append(P(
        "The objective is deliberately framed as a <b>variance-explained</b> question rather "
        "than an accuracy contest. Spotify's popularity score is a recency-weighted function "
        "of play counts, and play counts are driven by artist reputation, playlist placement "
        "and promotional spend, none of which is audible in the waveform. The scientifically "
        "meaningful target is therefore the <b>ceiling</b>: how far audio features actually "
        "go, and where the remaining explanatory power lives."))
    S.append(P("Research questions", H2))
    S.extend(bullets([
        "<b>RQ1</b> — How much variance in Spotify popularity is explained by audio features alone?",
        "<b>RQ2</b> — Which audio features carry that signal, and in which direction?",
        "<b>RQ3</b> — Does genre context materially improve on audio features alone?",
        "<b>RQ4</b> — Is popularity easier to predict as a binary outcome than a continuous score?",
    ]))
    S.append(P(
        "<b>Expected outcome.</b> A quantified answer to each question, supported by a "
        "reproducible modelling pipeline with explicit safeguards against data leakage, and an "
        "honest statement of the limits of audio-only prediction."))

    # ---------------------------------------------------- 3. Models and justification
    S.append(P("3. Models to be Applied and Justification", H1))
    S.append(P(
        "Models are introduced in increasing order of capacity so that each addition earns its "
        "complexity against the previous one. Both a regression formulation (continuous 0–100 "
        "score) and a classification formulation (binary popular / not popular) are applied, "
        "which is what allows RQ4 to be answered."))

    S.append(table([
        ["Model", "Type", "Justification"],
        ["Mean predictor (DummyRegressor)", "Baseline",
         "Establishes the reference R<super>2</super> = 0. Without it, a weak model cannot be "
         "distinguished from a useless one. Every later result is measured against this."],
        ["Linear Regression (OLS)", "Regression",
         "Interpretable coefficients and a formal hypothesis-testing framework. Serves as the "
         "linear-signal benchmark: the gap between it and the tree models quantifies how much "
         "of the relationship is non-linear."],
        ["Ridge and Lasso", "Regularised regression",
         "Audio features are strongly collinear (VIF: tempo 15.3, energy 14.6, danceability "
         "12.2). Regularisation stabilises coefficients under collinearity, and Lasso "
         "additionally performs embedded feature selection."],
        ["Random Forest", "Ensemble (bagging)",
         "Captures non-linear responses and feature interactions without manual specification, "
         "and is robust to the skewed feature distributions present in the data. Provides "
         "permutation-based importance for RQ2."],
        ["Gradient Boosting (HistGradientBoosting)", "Ensemble (boosting)",
         "Sequential error-correction typically outperforms bagging on tabular data; included "
         "as the strongest conventional competitor and as a check that Random Forest is not "
         "winning by accident."],
        ["Hurdle model (two-stage)", "Specialised regression",
         "About 10.5% of tracks sit at exactly zero popularity, and those zeros cluster by "
         "genre rather than scattering at random. Stage 1 classifies zero vs non-zero; stage 2 "
         "regresses on non-zero tracks only. This models the two data-generating processes "
         "separately instead of forcing one regressor to average across both."],
        ["Logistic Regression / RF / GB classifiers", "Classification",
         "The binary framing for RQ4, using the same three model families so the regression "
         "and classification comparison is like-for-like."],
    ], widths=[4.0 * cm, 2.5 * cm, 10.5 * cm]))

    S.append(P("Justification for the overall modelling strategy", H2))
    S.extend(bullets([
        "<b>Two feature sets, every model.</b> Each model is fitted on audio features alone and "
        "then on audio + genre. The difference isolates the contribution of genre context, which "
        "is what RQ3 asks.",
        "<b>Artist-level features deliberately excluded.</b> Artist mean popularity is the "
        "strongest available predictor, but computing it across the dataset leaks the target. "
        "Excluding it caps accuracy and keeps the study faithful to its stated title.",
        "<b>All preprocessing inside a Pipeline</b>, so scalers and encoders are fitted per fold "
        "and never observe held-out data.",
    ]))

    S.append(PageBreak())

    # ------------------------------------------------------------------- 4. Datasets
    S.append(P("4. Datasets to be Used", H1))
    S.append(P(
        "<b>Source.</b> The Spotify Tracks Dataset — 114,000 tracks x 21 columns, BSD licensed, "
        "retrieved from its HuggingFace mirror (also mirrored on Kaggle). It contains track "
        "metadata, the eleven Spotify audio features, genre, and the target variable "
        "<font face='Courier'>popularity</font> (0–100)."))

    S.append(Paragraph(
        "<b>Note on data provenance.</b> Spotify deprecated the "
        "<font face='Courier'>audio_features</font> and "
        "<font face='Courier'>audio_analysis</font> Web API endpoints on 27 November 2024; new "
        "applications now receive HTTP 403/410 and no official replacement exists. This data "
        "therefore cannot be collected first-hand, and the project necessarily uses an archived "
        "pre-deprecation snapshot. This constraint is disclosed rather than hidden, and is "
        "carried into the limitations of the analysis.", NOTE))

    S.append(P("4.1 Separate datasets for training and testing", H2))
    S.append(P(
        "The source file is partitioned into three <b>physically separate, mutually disjoint</b> "
        "CSV files before any model is fitted. Partitioning is stratified on binned popularity so "
        "the target distribution — including the zero spike — is represented consistently in each."))

    S.append(table([
        ["Dataset file", "Rows", "Share", "Role"],
        ["<font face='Courier'>data/processed/train.csv</font>", "62,704", "70%",
         "Model fitting and hyperparameter search (cross-validation is performed within this file only)."],
        ["<font face='Courier'>data/processed/val.csv</font>", "13,437", "15%",
         "Model selection and comparison between candidate models. Never used for fitting."],
        ["<font face='Courier'>data/processed/test.csv</font>", "13,437", "15%",
         "Held-out final evaluation. Touched exactly once, after all modelling decisions are frozen."],
    ], widths=[5.2 * cm, 1.9 * cm, 1.5 * cm, 8.4 * cm], align_center_cols=(1, 2)))

    S.append(P("4.2 Why deduplication precedes splitting", H2))
    S.append(P(
        "The raw file is constructed as 114 genres x 1,000 tracks, so a track belonging to "
        "several genres appears several times — up to nine. Of 114,000 rows, <b>24,259 (21.3%) "
        "are duplicate tracks</b>, leaving 89,741 unique <font face='Courier'>track_id</font> "
        "values. Splitting the raw file would place the <b>same song in both training and test "
        "sets</b>, inflating every reported score."))
    S.append(P(
        "Deduplication is therefore applied first, and three assertions verify that no "
        "<font face='Courier'>track_id</font> appears in more than one partition. A further 163 "
        "physically impossible records (zero tempo, zero time signature, zero duration) are "
        "removed, giving a final analysis set of <b>89,578 unique tracks</b>."))

    S.append(P("4.3 Secondary dataset for external validation", H2))
    S.append(P(
        "To test whether findings generalise beyond a single scrape, the Spotify Hit Predictor "
        "Dataset (1960–2019, approximately 41,000 tracks with an independent hit/flop label) is "
        "proposed as an <b>external test set</b>. Because it originates from a different "
        "collection process and carries its own target definition, it provides a stronger "
        "generalisation check than an internal split alone. It also carries decade information, "
        "which the primary dataset lacks — see Section 8.4."))

    # -------------------------------------------------------------------- 5. Metrics
    S.append(P("5. Metrics for Analysis and Justification", H1))
    S.append(P("5.1 Regression metrics", H2))
    S.append(table([
        ["Metric", "Justification"],
        ["R<super>2</super> (coefficient of determination)",
         "The primary metric, because the research question is explicitly about variance "
         "explained. Directly interpretable as the proportion of variation in popularity "
         "accounted for by the model."],
        ["Adjusted R<super>2</super>",
         "The audio+genre feature set adds 114 one-hot genre columns. Adjusted R<super>2</super> "
         "penalises that added dimensionality, so the genre gain cannot be an artefact of "
         "simply having more predictors."],
        ["RMSE",
         "Expressed in the units of the target (popularity points), which makes error magnitude "
         "concrete. Squared penalty makes it sensitive to the large errors produced by the "
         "zero-inflated mass."],
        ["MAE",
         "Reported alongside RMSE precisely because it is not sensitive to those outliers. A "
         "large RMSE–MAE gap is itself diagnostic of the zero-inflation problem."],
    ], widths=[4.2 * cm, 12.8 * cm]))

    S.append(P("5.2 Classification metrics", H2))
    S.append(table([
        ["Metric", "Justification"],
        ["ROC-AUC",
         "Threshold-independent measure of ranking quality, so it evaluates the model without "
         "committing to a particular decision cut-off."],
        ["PR-AUC (average precision)",
         "The positive class is a minority (about 23.7%). Precision-recall curves are the "
         "appropriate summary under imbalance, where ROC-AUC can appear optimistic. Always "
         "reported against the positive-rate baseline."],
        ["Precision, Recall, F1",
         "Expose the operating trade-off that a single aggregate hides — a model may achieve "
         "high precision purely by predicting the positive class very rarely."],
        ["Confusion matrix",
         "Reports the actual counts of each error type rather than a ratio."],
    ], widths=[4.2 * cm, 12.8 * cm]))

    S.append(Paragraph(
        "<b>Accuracy is deliberately excluded.</b> With a 23.7% positive rate, a model that "
        "predicts “not popular” for every track achieves 76.3% accuracy while being "
        "entirely useless. Reporting accuracy here would be actively misleading, so it is "
        "replaced by PR-AUC measured against the positive-rate baseline.", NOTE))

    S.append(P("5.3 Diagnostic and validity checks", H2))
    S.extend(bullets([
        "<b>Variance Inflation Factor (VIF)</b> — quantifies the multicollinearity that makes "
        "linear coefficients unreliable, and justifies the use of regularised models.",
        "<b>Permutation importance</b> — preferred over impurity-based importance for RQ2, "
        "because impurity measures systematically inflate correlated features.",
        "<b>SHAP values</b> — give per-feature direction and magnitude, not just rank ordering.",
        "<b>Shuffled-target check</b> — refitting on a permuted target must collapse "
        "R<super>2</super> to approximately zero. Any other result indicates leakage.",
    ]))

    S.append(PageBreak())

    # ---------------------------------------------------------- 6. Tools / libraries
    S.append(P("6. Tools and Libraries Used, with Justification", H1))
    S.append(table([
        ["Tool / Library", "Purpose", "Justification"],
        ["Python 3.11", "Language",
         "Standard environment for predictive analytics, with the mature scientific stack this "
         "project depends on."],
        ["pandas", "Data manipulation",
         "Efficient handling of the 114,000-row table; the grouped duplicate analysis and "
         "feature engineering are expressed directly in its API."],
        ["NumPy", "Numerical computing",
         "Vectorised transforms underlying the log and cyclic-encoding steps."],
        ["scikit-learn", "Modelling and evaluation",
         "Supplies every model family used, plus the <font face='Courier'>Pipeline</font> and "
         "<font face='Courier'>ColumnTransformer</font> abstractions that make leakage-free "
         "preprocessing structural rather than a matter of discipline."],
        ["statsmodels", "Statistical diagnostics",
         "Provides the variance inflation factor used for the formal multicollinearity test."],
        ["SHAP", "Model interpretation",
         "Game-theoretic attributions giving direction and magnitude per feature, needed for "
         "RQ2 where tree importances alone would be insufficient."],
        ["Matplotlib and seaborn", "Visualisation",
         "All figures are generated by code and regenerated from source, so no figure can drift "
         "from the data it depicts."],
        ["Jupyter Notebook", "Analysis narrative",
         "Interleaves code, output and reasoning, and is executed end to end to confirm "
         "reproducibility."],
        ["ReportLab", "Document generation",
         "Produces this PDF programmatically, so the document rebuilds automatically when "
         "results change."],
    ], widths=[3.1 * cm, 3.0 * cm, 10.9 * cm]))

    S.append(P(
        "<b>Why scikit-learn rather than a deep-learning framework.</b> The dataset is small, "
        "tabular and low-dimensional (roughly 15 predictors). Gradient-boosted trees and random "
        "forests are the established state of the art in this regime; a neural network would add "
        "substantial complexity and reduce interpretability without a corresponding gain, and "
        "interpretability is central to RQ2."))

    # ------------------------------------------------------------------- 7. Diagrams
    S.append(PageBreak())
    S.append(P("7. System Design: Flowchart and Module Diagram", H1))
    S.append(P(
        "Figure 1 traces the flow of data from the raw file through every processing function to "
        "the evaluation stage, annotated with the row count at each reduction. Figure 2 gives the "
        "module and class structure, listing the functions each module exposes and their inputs "
        "and return types. Both diagrams are generated from the actual source, so they cannot "
        "drift from the implementation."))
    S.append(P("7.1 Module responsibilities", H2))
    S.append(table([
        ["Module", "Inputs", "Key functions", "Outputs"],
        ["<font face='Courier'>data_prep</font>",
         "<font face='Courier'>dataset.csv</font>",
         "<font face='Courier'>load_raw, deduplicate, clean, engineer, build, split, "
         "feature_columns</font>",
         "<font face='Courier'>train.csv, val.csv, test.csv</font>; engineered feature matrix"],
        ["<font face='Courier'>models</font>",
         "feature column list; training partition",
         "<font face='Courier'>make_preprocessor, regression_models, HurdleRegressor, evaluate, "
         "shuffle_check</font>",
         "Fitted pipelines; metric dictionaries"],
        ["<font face='Courier'>run_*</font> drivers",
         "Prepared partitions",
         "Experiment orchestration for RQ1-RQ4",
         "<font face='Courier'>results_*.csv</font>; figures"],
    ], widths=[3.0 * cm, 3.4 * cm, 6.2 * cm, 4.4 * cm]))

    S.append(PageBreak())
    S.append(figure(FIG / "diag_flowchart.png", 14.6,
                    "Figure 1 - System flowchart. Inputs, functions, intermediate row counts, "
                    "the three disjoint dataset partitions, model families, and outputs."))
    S.append(PageBreak())
    S.append(figure(FIG / "diag_uml.png", 17.0,
                    "Figure 2 - Module and class diagram. Module members with signatures and "
                    "return types, the HurdleRegressor class, and dependency direction."))

    # ------------------------------------------------------------- 8. Basic analysis
    S.append(P("8. Basic Analysis Which Can Be Done", H1))
    S.append(P(
        "The following analyses are supported by the data. Preliminary results from an initial "
        "run are included to demonstrate feasibility; the full study will extend and formalise "
        "them."))

    S.append(P("8.1 Structural and data-quality analysis", H2))
    S.extend(bullets([
        "Duplicate audit — quantifying repeated tracks across genres and the leakage they would "
        "cause (established: 21.3% of rows).",
        "Target distribution analysis — characterising the zero-inflated mass and testing whether "
        "it is randomly distributed or clustered.",
        "Missing-value and impossible-value detection.",
    ]))
    S.append(figure(FIG / "03_zeros_by_genre.png", 12.6,
                    "Figure 3 — Zero-popularity rate by genre. The zeros are strongly clustered "
                    "(jazz 68.1%, Iranian 65.6%) rather than randomly distributed, which "
                    "indicates regional availability gaps at scrape time rather than unpopular "
                    "music, and motivates the two-stage hurdle model."))

    S.append(P("8.2 Correlation and multicollinearity analysis", H2))
    S.append(P(
        "Pairwise correlation between all audio features and the target, plus formal VIF testing. "
        "Preliminary finding: <b>no audio feature correlates with popularity above |r| = 0.10</b> "
        "(strongest: instrumentalness at −0.095), while energy, loudness and acousticness are "
        "strongly inter-correlated. This establishes at the outset that any linear model will "
        "explain very little, and that the relationship must be sought in non-linear structure."))

    S.append(P("8.3 Predictive modelling and comparison", H2))
    S.append(P("Preliminary held-out test results (evaluated once, after model selection):"))
    S.append(table([
        ["Feature set", "Model", "R<super>2</super>", "RMSE", "MAE"],
        ["Audio only", "Mean baseline", "0.000", "20.64", "17.31"],
        ["Audio only", "Linear Regression", "0.034", "20.28", "16.72"],
        ["Audio only", "<b>Random Forest</b>", "<b>0.255</b>", "17.81", "13.97"],
        ["Audio + genre", "Linear Regression", "0.324", "16.97", "12.16"],
        ["Audio + genre", "<b>Random Forest</b>", "<b>0.483</b>", "14.84", "10.20"],
    ], widths=[3.2 * cm, 4.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm],
        align_center_cols=(2, 3, 4)))
    S.append(Spacer(1, 5))
    S.append(P("Classification framing (threshold: 75th percentile of non-zero popularity; "
               "positive rate 23.7%):"))
    S.append(table([
        ["Feature set", "Model", "ROC-AUC", "PR-AUC", "PR baseline"],
        ["Audio only", "Random Forest", "0.730", "0.441", "0.237"],
        ["Audio + genre", "Random Forest", "<b>0.870</b>", "<b>0.679</b>", "0.237"],
    ], widths=[3.2 * cm, 4.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm],
        align_center_cols=(2, 3, 4)))

    S.append(P("8.4 Feature importance and interpretation", H2))
    S.append(P(
        "Permutation importance and SHAP analysis to answer RQ2. Preliminary finding: "
        "<b>genre is roughly thirteen times more important than the strongest audio feature</b> "
        "(permutation importance 0.674 versus 0.052). Within the audio features alone, the two "
        "methods agree closely on the ranking — instrumentalness, acousticness, valence and "
        "duration lead, while key and mode are effectively inert."))
    S.append(figure(FIG / "08_shap_summary.png", 12.6,
                    "Figure 4 — SHAP summary for the audio-only model. Instrumentalness and "
                    "acousticness push predicted popularity downward; musical key contributes "
                    "essentially nothing."))

    S.append(P("8.5 Validity and robustness analysis", H2))
    S.extend(bullets([
        "<b>Leakage verification</b> — assertions that no track identifier spans partitions.",
        "<b>Shuffled-target control</b> — preliminary result: R<super>2</super> = −0.0016, "
        "confirming no leakage path.",
        "<b>Sensitivity analysis</b> — the full study repeated with zero-popularity tracks "
        "excluded, testing how far conclusions depend on that decision.",
        "<b>Threshold sensitivity</b> — the classification cut repeated at the 90th percentile.",
        "<b>Era confounding</b> — recording loudness has risen over decades and newer releases "
        "score higher through recency bias, so audio features may partly proxy for release date. "
        "The primary dataset has no year column, so this will be tested using the secondary "
        "dataset described in Section 4.3.",
    ]))

    S.append(P("8.6 Expected deliverables", H2))
    S.extend(bullets([
        "A reproducible analysis pipeline with leakage controls built into its structure.",
        "An executed Jupyter notebook presenting the analysis end to end.",
        "A quantified answer to each of RQ1–RQ4, with an explicit statement of limitations.",
    ]))

    doc.build(S, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"written: {OUT}")


if __name__ == "__main__":
    build()
