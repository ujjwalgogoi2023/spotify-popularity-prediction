"""Generate the flowchart and UML module diagram for the DA1 document.

Layout is computed rather than hardcoded: every box derives its height from its line
count, so text can never overflow its container. Content mirrors the actual signatures
in src/data_prep.py and src/models.py.

Coordinate system: x in [0, 100]; y in "line units" counting down from the top. The
figure height is chosen so one line unit maps to a fixed point size.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INPUT   = "#E3EDFB"
PROCESS = "#FFFFFF"
MODEL   = "#FDF0E0"
OUTPUT  = "#E4F5EA"
EXTERN  = "#F2F4F5"
EDGE    = "#33404F"
ACCENT  = "#B03A2E"

LINE  = 1.0    # one body line
TITLE = 1.5    # title band
PAD   = 0.45   # internal padding, top and bottom
GAP   = 1.5    # vertical gap between stacked boxes


def box_height(n_lines: int) -> float:
    return PAD + TITLE + n_lines * LINE + PAD


class Canvas:
    def __init__(self, total_lines, width_in=9.4, unit_pt=13.0):
        self.total = total_lines
        self.unit_pt = unit_pt
        self.width_in = width_in
        height_in = total_lines * unit_pt / 72.0
        self.fig, self.ax = plt.subplots(figsize=(width_in, height_in))
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(total_lines, 0)   # y grows downward
        self.ax.axis("off")

    def finalize(self, max_y, margin=1.5):
        """Resize to the layout actually produced, so nothing is clipped."""
        total = max_y + margin
        self.ax.set_ylim(total, 0)
        self.fig.set_size_inches(self.width_in, total * self.unit_pt / 72.0)

    def box(self, x, y, w, title, lines, fc=PROCESS, title_size=9, body_size=7.4):
        """Draw a box whose top-left is (x, y). Returns its height."""
        h = box_height(len(lines))
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.35",
            linewidth=1.15, edgecolor=EDGE, facecolor=fc, zorder=2))
        self.ax.text(x + w / 2, y + PAD + TITLE * 0.72, title, ha="center", va="center",
                     fontsize=title_size, fontweight="bold", color="#17202A", zorder=3)
        for i, ln in enumerate(lines):
            self.ax.text(x + w / 2, y + PAD + TITLE + (i + 0.5) * LINE, ln,
                         ha="center", va="center", fontsize=body_size,
                         family="DejaVu Sans Mono", color="#2C3E50", zorder=3)
        return h

    def arrow(self, p, q, color=EDGE):
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12,
                                          linewidth=1.2, color=color, zorder=1,
                                          shrinkA=0, shrinkB=0))

    def note(self, x, y, text, color=ACCENT, size=7.2, ha="left"):
        self.ax.text(x, y, text, fontsize=size, color=color, ha=ha, va="center", zorder=4)

    def title(self, main, sub):
        self.ax.text(50, 0.85, main, ha="center", va="center",
                     fontsize=12, fontweight="bold", color="#17202A")
        self.ax.text(50, 2.05, sub, ha="center", va="center",
                     fontsize=8, style="italic", color="#5D6D7E")

    def save(self, path):
        self.fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(self.fig)


def flowchart(path):
    c = Canvas(total_lines=64)
    c.title("Figure 1 — System Flowchart: Data Pipeline and Modelling Stage",
            "Function names correspond exactly to src/data_prep.py and src/models.py")

    CX, W = 28.0, 44.0     # centred column
    y = 3.6

    steps = [
        ("INPUT   data/raw/dataset.csv",
         ["114,000 rows x 21 columns", "Spotify Tracks Dataset (BSD licence)"], INPUT, None),
        ("load_raw()",
         ["pd.read_csv(index_col=0)", "returns: DataFrame"], PROCESS, None),
        ("deduplicate(df)",
         ["n_genres = groupby(track_id).nunique()", "drop_duplicates(subset='track_id')"],
         PROCESS, "114,000  ->  89,741\n(-24,259 duplicate rows, 21.3%)"),
        ("clean(df)",
         ["drop tempo == 0, time_signature == 0,", "duration_ms == 0"],
         PROCESS, "89,741  ->  89,578\n(-163 impossible records)"),
        ("engineer(df)",
         ["log1p: speechiness, liveness,",
          "       instrumentalness, acousticness",
          "key -> key_sin, key_cos  (cyclic)",
          "duration_ms -> log_duration"], PROCESS, None),
        ("build(exclude_zeros=False)",
         ["orchestrates the three steps above",
          "True -> Track B sensitivity analysis"], EXTERN, None),
        ("split(df, random_state=42)",
         ["stratified on binned popularity",
          "asserts no track_id spans partitions"], PROCESS, None),
    ]

    for title, lines, fc, note in steps:
        h = c.box(CX, y, W, title, lines, fc=fc)
        if note:
            c.note(CX + W + 2.5, y + h / 2, note)
        c.arrow((50, y + h), (50, y + h + GAP))
        y += h + GAP

    # --- three disjoint partitions -------------------------------------------------
    split_y = y
    parts = [("train.csv", "62,704 tracks  (70%)", 2.0),
             ("val.csv",   "13,437 tracks  (15%)", 35.5),
             ("test.csv",  "13,437 tracks  (15%)", 69.0)]
    ph = 0
    for name, sub, px in parts:
        ph = c.box(px, split_y, 29.0, name, [sub], fc=INPUT)
        c.arrow((50, split_y - GAP), (px + 14.5, split_y))
    # No caption here: split() already states the disjointness guarantee, and a label
    # in this band collides with the fan-out arrows.

    y = split_y + ph + GAP + 1.0

    # All three partitions feed the same column-selection and preprocessing stage;
    # the fit/transform asymmetry lives inside the Pipeline, not in the data flow.
    hf = c.box(CX, y, W, "feature_columns(include_genre)",
               ["False -> 15 audio features  (RQ1)",
                "True  -> + track_genre, n_genres  (RQ3)"], fc=PROCESS)
    for _, _, px in parts:
        c.arrow((px + 14.5, split_y + ph), (50, y))
    y += hf + GAP

    hp = c.box(CX, y, W, "make_preprocessor(cols)",
               ["StandardScaler  |  OneHotEncoder",
                "fit on TRAIN fold only, inside Pipeline"], fc=MODEL)
    c.arrow((50, y - GAP), (50, y))
    y += hp + GAP

    # --- model families -------------------------------------------------------------
    ml = c.box(2.0, y, 45.0, "regression_models(cols)",
               ["DummyRegressor  (mean baseline)",
                "LinearRegression / Ridge / Lasso",
                "RandomForestRegressor",
                "HistGradientBoostingRegressor"], fc=MODEL)
    mr = c.box(53.0, y, 45.0, "HurdleRegressor(cols)",
               ["stage 1:  P(popularity > 0)",
                "stage 2:  E[pop | pop > 0]",
                "predict = p * conditional",
                "guards single-class targets"], fc=MODEL)
    c.arrow((50, y - GAP), (24.5, y))
    c.arrow((50, y - GAP), (75.5, y))
    y += max(ml, mr) + GAP

    # --- evaluation -----------------------------------------------------------------
    he = c.box(14.0, y, 72.0, "evaluate()   /   shuffle_check()",
               ["R2, Adjusted R2, RMSE, MAE   |   ROC-AUC, PR-AUC, F1",
                "shuffle_check: permuted target must collapse R2 to ~0"],
               fc=PROCESS, body_size=7.2)
    c.arrow((24.5, y - GAP), (40.0, y))
    c.arrow((75.5, y - GAP), (60.0, y))
    y += he + GAP

    ho = c.box(14.0, y, 72.0, "OUTPUT",
               ["data/processed/results_*.csv   |   figures/*.png   |   report/"],
               fc=OUTPUT, body_size=7.2)
    c.arrow((50, y - GAP), (50, y))

    c.finalize(y + ho)
    c.save(path)


def uml(path):
    c = Canvas(total_lines=46, width_in=10.0)
    c.title("Figure 2 — Module and Class Diagram (UML style)",
            "Members extracted from source; open arrows denote dependency (import / uses)")

    SEP = "-" * 38

    h1 = c.box(2.0, 4.0, 45.0, "<<module>>  data_prep", [
        "+ AUDIO_FEATURES : list[str]",
        "+ SKEWED : list[str]",
        "+ RANDOM_STATE : int = 42",
        SEP,
        "+ load_raw() -> DataFrame",
        "+ deduplicate(df) -> DataFrame",
        "+ clean(df) -> DataFrame",
        "+ engineer(df) -> DataFrame",
        "+ feature_columns(include_genre) -> list",
        "+ build(exclude_zeros) -> DataFrame",
        "+ split(df, random_state)",
        "      -> (train, val, test)",
    ], fc=INPUT, body_size=7.2)

    h2 = c.box(53.0, 4.0, 45.0, "<<module>>  models", [
        "+ make_preprocessor(feature_cols)",
        "      -> ColumnTransformer",
        "+ regression_models(feature_cols)",
        "      -> dict[str, Pipeline]",
        "+ evaluate(y_true, y_pred, n_features)",
        "      -> dict",
        "+ shuffle_check(model, X_train, y_train,",
        "      X_val, y_val) -> float",
        SEP,
        "class HurdleRegressor",
        "  - clf : Pipeline   (stage 1)",
        "  - reg : Pipeline   (stage 2)",
        "  + fit(X, y) -> self",
        "  + predict(X) -> ndarray",
    ], fc=MODEL, body_size=7.2)

    c.arrow((47.0, 4.0 + h1 / 2), (53.0, 4.0 + h1 / 2))
    c.note(50.0, 4.0 + h1 / 2 - 1.0, "imports", size=7.0, ha="center")

    y2 = 4.0 + max(h1, h2) + 3.0

    h3 = c.box(2.0, y2, 30.0, "<<driver>>  run_* scripts", [
        "run_regression.py      RQ1, RQ3",
        "run_classification.py  RQ4",
        "run_interpretation.py  EDA, VIF",
        "run_shap.py            SHAP",
        "run_final_test.py      held-out test",
        "make_notebook.py       notebook",
        "make_diagrams.py       figures 1-2",
    ], fc=PROCESS, body_size=7.2)

    # Artifacts sit next to their producer (the drivers); the external library sits
    # under models, which is what actually consumes it.
    h4 = c.box(36.0, y2, 29.0, "<<artifact>>  outputs", [
        "data/processed/train|val|test.csv",
        "data/processed/results_*.csv",
        "figures/01-08 + diagrams",
        "notebooks/*.ipynb",
        "report/DA1_report.pdf",
    ], fc=OUTPUT, body_size=7.2)

    h5 = c.box(69.0, y2, 29.0, "<<external>>  scikit-learn", [
        "Pipeline, ColumnTransformer",
        "StandardScaler, OneHotEncoder",
        "RandomForestRegressor",
        "HistGradientBoostingRegressor",
        "permutation_importance",
    ], fc=EXTERN, body_size=7.2)

    c.arrow((17.0, y2), (17.0, 4.0 + h1))
    c.note(18.2, y2 - 1.5, "imports", size=7.0)
    c.arrow((32.0, y2 + h3 / 2), (36.0, y2 + h3 / 2))
    c.note(34.0, y2 + h3 / 2 - 1.3, "produces", size=7.0, ha="center")
    c.arrow((83.5, 4.0 + h2), (83.5, y2))
    c.note(84.7, y2 - 1.5, "uses", size=7.0)

    c.finalize(y2 + max(h3, h4, h5))
    c.save(path)


if __name__ == "__main__":
    flowchart("figures/diag_flowchart.png")
    uml("figures/diag_uml.png")
    print("wrote figures/diag_flowchart.png and figures/diag_uml.png")
