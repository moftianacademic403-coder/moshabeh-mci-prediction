"""Generate all figures embedded in the final manuscript and supplement.

This script is executed as the final cell of the Moshabeh Kaggle notebook. It
uses only objects and locked outputs created earlier in that notebook. It does
not refit, retune, or recalibrate any prediction model.
"""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from PIL import Image, ImageDraw, ImageFont
import shap
import statsmodels.api as sm
from patsy import dmatrix, build_design_matrices
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(globals()["ROOT"])
FIG_MAIN = Path(globals()["FIG_MAIN"])
FIG_SUPP = Path(globals()["FIG_SUPP"])
TABLES = Path(globals()["TABLES"])
OUT = ROOT / "submission_figure_bundle"
MAIN_OUT = OUT / "main_figures"
SUPP_OUT = OUT / "supplementary_figures"
if OUT.exists():
    shutil.rmtree(OUT)
MAIN_OUT.mkdir(parents=True, exist_ok=True)
SUPP_OUT.mkdir(parents=True, exist_ok=True)

champion = globals()["champion"]
y_dev = np.asarray(globals()["y_dev"], dtype=int)
y_ext = np.asarray(globals()["y_ext"], dtype=int)
oof = globals()["oof"]
external_predictions = globals()["external_predictions"]
thresholds = globals()["thresholds"]
net_benefit = globals()["net_benefit"]
missingness = globals()["missingness"]
importance = globals()["importance"].copy()
rank_corr = globals()["rank_corr"].copy()
explanation = globals()["explanation"]
X_shap = np.asarray(globals()["X_shap"])
transformed_names = np.asarray(globals()["transformed_names"], dtype=str)
sample = globals()["sample"]


BLUE = "#2C78B8"
ORANGE = "#D55E00"
INK = "#26313C"
GREY = "#555555"
PALETTE_LOCAL = list(plt.get_cmap("tab10").colors)


DISPLAY = {
    "cat__EDU": "Education",
    "num__age": "Age",
    "cat__sleep_quality": "Sleep quality",
    "cat__iadl": "IADL",
    "num__visual_acuity_both": "Binocular visual acuity",
    "cat__household_income": "Household income",
    "cat__missingindicator_iadl": "IADL missing (indicator)",
    "num__waist_hip_ratio": "Waist-to-hip ratio",
    "cat__adl": "ADL",
    "cat__employment_status": "Employment status",
    "cat__depression_status": "Depression status",
    "num__whisper_test_left": "Whisper test, left ear",
    "num__whisper_test_right": "Whisper test, right ear",
    "num__Albumin": "Albumin",
    "num__BMI": "Body-mass index",
    "cat__osteoporosis_status": "Osteoporosis",
    "cat__hypertension": "Hypertension",
    "num__max_handgrip": "Maximum handgrip strength",
    "cat__sex": "Sex",
    "EDU": "Education",
    "iadl": "IADL",
    "visual_acuity_both": "Binocular visual acuity",
    "household_income": "Household income",
    "sleep_quality": "Sleep quality",
    "employment_status": "Employment status",
    "max_handgrip": "Maximum handgrip strength",
    "depression_status": "Depression status",
    "adl": "ADL",
    "BMI": "BMI",
    "osteoporosis_status": "Osteoporosis status",
    "age": "Age",
}


def display_name(raw: str) -> str:
    raw = str(raw).strip()
    return DISPLAY.get(
        raw,
        raw.replace("cat__", "").replace("num__", "").replace("_", " ").capitalize(),
    )


def save_figure(fig, folder: Path, stem: str, dpi: int = 600) -> None:
    fig.savefig(folder / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_pil(image: Image.Image, folder: Path, stem: str, dpi: int = 300) -> None:
    image.save(folder / f"{stem}.png", dpi=(dpi, dpi), optimize=True)
    image.save(folder / f"{stem}.pdf", "PDF", resolution=dpi)


def as_probability_array(frame, name: str) -> np.ndarray:
    value = frame[name]
    return value.to_numpy(dtype=float) if hasattr(value, "to_numpy") else np.asarray(value, dtype=float)


def performance_panel(y, predictions, stem, folder, suptitle=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_roc, ax_cal, ax_pr, ax_dca = axes.ravel()
    grid = np.linspace(0.05, 0.50, 100)
    prevalence = float(np.mean(y))
    for i, (name, p) in enumerate(predictions.items()):
        p = np.asarray(p, dtype=float)
        color = PALETTE_LOCAL[i % len(PALETTE_LOCAL)]
        fpr, tpr, _ = roc_curve(y, p)
        ax_roc.plot(fpr, tpr, lw=1.7, color=color,
                    label=f"{name} ({roc_auc_score(y, p):.3f})")
        bins = pd.qcut(p, q=min(10, len(np.unique(p))), duplicates="drop")
        cal = (pd.DataFrame({"y": y, "p": p, "bin": bins})
               .groupby("bin", observed=True)
               .agg(observed=("y", "mean"), predicted=("p", "mean")))
        ax_cal.plot(cal.predicted, cal.observed, "o-", lw=1.4, color=color, label=name)
        precision, recall, _ = precision_recall_curve(y, p)
        ax_pr.plot(recall, precision, lw=1.7, color=color,
                   label=f"{name} ({average_precision_score(y, p):.3f})")
        ax_dca.plot(grid, net_benefit(y, p, grid), lw=1.5, color=color, label=name)
    ax_roc.plot([0, 1], [0, 1], "--", color="gray")
    ax_roc.set(xlabel="1 - Specificity", ylabel="Sensitivity", title="A. ROC curves")
    ax_cal.plot([0, 1], [0, 1], "--", color="gray")
    ax_cal.set(xlabel="Predicted probability", ylabel="Observed proportion", title="B. Calibration")
    ax_pr.axhline(prevalence, ls="--", color="gray")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="C. Precision-recall curves")
    treat_all = prevalence - (1 - prevalence) * grid / (1 - grid)
    ax_dca.plot(grid, treat_all, "--", color="black", label="Screen all")
    ax_dca.axhline(0, ls=":", color="gray", label="Screen none")
    if champion in predictions:
        locked = float(thresholds[champion])
        nb = float(net_benefit(y, np.asarray(predictions[champion]), [locked])[0])
        ax_dca.axvline(locked, ls="-.", lw=1.3, color=ORANGE,
                       label=f"Locked threshold ({locked:.3f})")
        ax_dca.scatter([locked], [nb], s=38, color=ORANGE, zorder=5)
    ax_dca.set(xlabel="Threshold probability", ylabel="Net benefit",
               title="D. Decision curve analysis", xlim=(0.05, 0.50))
    for ax in axes.ravel():
        ax.legend(frameon=False, fontsize=7)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97] if suptitle else None)
    save_figure(fig, folder, stem, dpi=600)


def figure2_and_s2():
    model_order = [name for name in globals()["MODEL_SPECS"]]
    ext_preds = {name: as_probability_array(external_predictions, name) for name in model_order}
    performance_panel(y_ext, ext_preds, "Figure2_External_validation_performance", MAIN_OUT)
    dev_preds = {
        "DemographicLR": as_probability_array(oof, "DemographicLR"),
        champion: as_probability_array(oof, champion),
    }
    performance_panel(
        y_dev, dev_preds, "Supplementary_Figure_S2_Development_baseline_vs_selected_model",
        SUPP_OUT, "Supplementary Figure S2. Small baseline versus selected model",
    )


def figure3_hexbin():
    models = [("DemographicLR", "Demographic baseline"),
              ("ElasticNet", "Elastic-net logistic regression")]
    cmap = LinearSegmentedColormap.from_list(
        "count_blue", ["#DCE8F3", "#92BAD8", "#397DAD", "#174F7A", "#092D49"]
    )
    tab = as_probability_array(external_predictions, champion)
    fig = plt.figure(figsize=(190 / 25.4, 98 / 25.4))
    gs = fig.add_gridspec(1, 3, left=0.085, right=0.925, bottom=0.235, top=0.825,
                          width_ratios=[1, 1, 0.045], wspace=0.23)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    collections = []
    for i, (ax, (key, title)) in enumerate(zip(axes, models)):
        comp = as_probability_array(external_predictions, key)
        coll = ax.hexbin(comp, tab, gridsize=(28, 16), extent=(0, 1, 0, 1),
                         mincnt=1, cmap=cmap, bins="log", linewidths=0.16,
                         edgecolors="face", zorder=2)
        collections.append(coll)
        ax.plot([0, 1], [0, 1], color="#333333", linewidth=0.85,
                linestyle=(0, (4, 3)), zorder=3)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
               xlabel="Comparator predicted probability")
        ax.set_xticks(np.linspace(0, 1, 6)); ax.set_yticks(np.linspace(0, 1, 6))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_title(title, loc="left", pad=8)
        ax.text(-0.015, 1.105, chr(65 + i), transform=ax.transAxes,
                fontweight="bold", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(False)
    axes[0].set_ylabel("TabPFN predicted probability")
    cb = fig.colorbar(collections[0], cax=cax)
    cb.set_label("Participants per hexagon (log scale)", fontsize=7.5, labelpad=8)
    fig.legend(handles=[Line2D([0], [0], color="#333333", linewidth=0.85,
                               linestyle=(0, (4, 3)), label="Equal predicted probabilities")],
               loc="lower center", bbox_to_anchor=(0.47, 0.055), frameon=False,
               handlelength=2.6, fontsize=7.5)
    save_figure(fig, MAIN_OUT, "Figure3_Prediction_agreement_hexbin", dpi=600)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def flexible_calibration(y, p, cohort, spline_df=4):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lp = _logit(p)
    design = dmatrix(f"cr(lp, df={spline_df})", {"lp": lp}, return_type="dataframe")
    model = sm.GLM(y, design, family=sm.families.Binomial()).fit()
    grid = np.linspace(p.min(), p.max(), 300)
    grid_design = build_design_matrices([design.design_info], {"lp": _logit(grid)})[0]
    sf = model.get_prediction(grid_design).summary_frame(alpha=0.05)
    fitted = np.asarray(model.predict(design))
    error = np.abs(fitted - p)
    bins = pd.qcut(p, q=min(10, len(np.unique(p))), duplicates="drop")
    dec = (pd.DataFrame({"y": y, "p": p, "bin": bins})
           .groupby("bin", observed=True)
           .agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
           .reset_index(drop=True))
    return grid, sf, dec, {"ICI": float(error.mean()), "E90": float(np.quantile(error, 0.90))}


def figure4_calibration():
    analyses = [
        ("Nested OOF", y_dev, as_probability_array(oof, champion)),
        ("External locked", y_ext, as_probability_array(external_predictions, champion)),
    ]
    palette = {"Nested OOF": "#0072B2", "External locked": ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    for ax, (name, y, p) in zip(axes, analyses):
        grid, sf, dec, summary = flexible_calibration(y, p, name)
        color = palette[name]
        ax.fill_between(grid, sf["mean_ci_lower"].to_numpy(), sf["mean_ci_upper"].to_numpy(),
                        color=color, alpha=0.18, linewidth=0, label="Pointwise 95% CI")
        ax.plot(grid, sf["mean"].to_numpy(), color=color, linewidth=2.2,
                label="Flexible spline curve")
        ax.scatter(dec["predicted"], dec["observed"],
                   s=np.clip(dec["n"] * 0.35, 35, 95), facecolor="white",
                   edgecolor=color, linewidth=1.4, zorder=4, label="Risk deciles")
        ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1.1,
                label="Ideal calibration")
        inset = ax.inset_axes([0.08, 0.04, 0.86, 0.15])
        inset.hist(p, bins=25, color=color, alpha=0.28)
        inset.set_xlim(0, 1); inset.set_xticks([]); inset.set_yticks([])
        for spine in inset.spines.values(): spine.set_visible(False)
        inset.set_facecolor("none")
        ax.set_title(f"{name}\nICI={summary['ICI']:.3f}; E90={summary['E90']:.3f}")
        ax.set_xlabel("Predicted probability of screen-positive cognitive impairment")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Observed probability of screen-positive cognitive impairment")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_figure(fig, MAIN_OUT, "Figure4_Flexible_calibration", dpi=600)


def fig_to_image(fig, dpi=300):
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB")


def fit_image(img, box_w, box_h):
    scale = min(box_w / img.width, box_h / img.height)
    return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)


FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
F_REG = str(FONT_DIR / "DejaVuSans.ttf")
F_BOLD = str(FONT_DIR / "DejaVuSans-Bold.ttf")


def pil_font(size, bold=False):
    return ImageFont.truetype(F_BOLD if bold else F_REG, size)


def paste_panel(canvas, img, box, letter, title, letter_size=58, title_size=42):
    x, y, w, h = box
    draw = ImageDraw.Draw(canvas)
    head = max(letter_size, title_size) + 24
    draw.text((x, y), letter, font=pil_font(letter_size, True), fill="black")
    draw.text((x + letter_size + 16, y + 8), title, font=pil_font(title_size), fill="black")
    fitted = fit_image(img, w, h - head)
    canvas.paste(fitted, (x + (w - fitted.width) // 2, y + head))


def panel_shap_beeswarm():
    values = np.asarray(explanation.values)
    top_idx = np.argsort(np.mean(np.abs(values), axis=0))[::-1][:19]
    exp_top = explanation[:, top_idx]
    exp_top.feature_names = [display_name(transformed_names[i]) for i in top_idx]
    plt.figure(figsize=(8.2, 10.5))
    shap.plots.beeswarm(exp_top, max_display=19, show=False, plot_size=None)
    plt.xlabel("SHAP value (impact on model output)")
    plt.title("")
    return fig_to_image(plt.gcf(), dpi=300)


def panel_subgroups():
    rows = pd.read_csv(TABLES / "S5_external_subgroup_performance.csv")
    pretty = {
        ("education", "illiterate_or_primary"): "Illiterate / primary",
        ("education", "secondary_or_academic"): "Secondary / academic",
        ("age", "<70"): "Age < 70 y", ("age", ">=70"): "Age ≥ 70 y",
        ("sex", "female"): "Women", ("sex", "male"): "Men",
    }
    order = list(pretty)
    idx = {(r.variable, r.group): r for r in rows.itertuples()}
    overall = roc_auc_score(y_ext, as_probability_array(external_predictions, champion))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ypos = list(range(len(order)))[::-1]
    for y, key in zip(ypos, order):
        r = idx[key]; est, lo, hi = float(r.AUROC), float(r.AUROC_CI_low), float(r.AUROC_CI_high)
        color = ORANGE if key[0] == "education" else BLUE
        ax.plot([lo, hi], [y, y], color=color, lw=2.4, solid_capstyle="round")
        ax.plot([est], [y], "o", color=color, ms=8)
        ax.text(1.005, y, f"{est:.3f} ({lo:.3f}–{hi:.3f})", va="center",
                fontsize=9.5, transform=ax.get_yaxis_transform())
    ax.axvline(overall, color=GREY, ls="--", lw=1.5,
               label=f"Overall external AUROC ({overall:.3f})")
    ax.set_yticks(ypos, [f"{pretty[k]}  (n={int(idx[k].n)})" for k in order], fontsize=10.5)
    ax.set_xlim(0.55, 1.0); ax.set_xlabel("AUROC (95% bootstrap CI)")
    ax.grid(axis="x", alpha=0.35); ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    return fig_to_image(fig)


def panel_education():
    rows = pd.read_csv(TABLES / "S6_education_sensitivity.csv").set_index("analysis")
    settings = [
        ("Development\n(nested OOF)", "With education — nested OOF", "Without education — nested OOF"),
        ("External\nvalidation", "With education — external", "Without education — external"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for x, (label, full_key, no_key) in enumerate(settings):
        full = float(rows.loc[full_key, "AUROC"]); noedu = float(rows.loc[no_key, "AUROC"])
        ax.plot([x - 0.13, x + 0.13], [full, noedu], color=GREY, lw=1.6)
        ax.plot(x - 0.13, full, "o", color=BLUE, ms=11)
        ax.plot(x + 0.13, noedu, "o", color=ORANGE, ms=11)
        ax.annotate(f"{full:.3f}", (x - 0.13, full), xytext=(0, 12),
                    textcoords="offset points", ha="center", color=BLUE)
        ax.annotate(f"{noedu:.3f}", (x + 0.13, noedu), xytext=(0, -20),
                    textcoords="offset points", ha="center", color=ORANGE)
        ax.annotate(f"Δ = {noedu - full:+.3f}", (x, (full + noedu) / 2),
                    xytext=(26, -3), textcoords="offset points", ha="left")
    ax.set_xticks([0, 1], [x[0] for x in settings]); ax.set_ylabel("AUROC")
    ax.set_xlim(-0.45, 1.55); ax.set_ylim(0.782, 0.883)
    ax.grid(axis="y", alpha=0.35); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="Full predictor set"),
                       Line2D([], [], marker="o", ls="", color=ORANGE, label="Education removed")],
              loc="upper left", frameon=False)
    return fig_to_image(fig)


def figure5_shap_subgroups():
    left = panel_shap_beeswarm(); top = panel_subgroups(); bottom = panel_education()
    width, height, margin, gap = 4400, 2920, 110, 110
    canvas = Image.new("RGB", (width, height), "white")
    body_top = 60
    left_w = int((width - 2 * margin - gap) * 0.46)
    right_w = width - 2 * margin - gap - left_w
    body_h = height - body_top - margin
    paste_panel(canvas, left, (margin, body_top, left_w, body_h), "A",
                "SHAP summary — TabPFN, external cohort")
    rh = (body_h - gap) // 2
    paste_panel(canvas, top, (margin + left_w + gap, body_top, right_w, rh), "B",
                "External discrimination by subgroup")
    paste_panel(canvas, bottom, (margin + left_w + gap, body_top + rh + gap, right_w, rh), "C",
                "Sensitivity to removing education")
    save_pil(canvas, MAIN_OUT, "Figure5_SHAP_subgroups_education", dpi=300)


def figure6_importance():
    models = ["ElasticNet", "SVM", "RF", "GBM", "XGBoost", "CatBoost", "LightGBM", "TabPFN"]
    all_models = ["DemographicLR", *models]
    ranks = importance[models].rank(ascending=False, method="average")
    order = pd.DataFrame({"median_rank": ranks.median(axis=1), "mean_rank": ranks.mean(axis=1)})
    order["feature"] = order.index
    top = order.sort_values(["median_rank", "mean_rank", "feature"]).head(12).index.tolist()
    subcorr = rank_corr.loc[models, models]
    labels = DISPLAY
    model_labels = {"DemographicLR": "Demographic baseline", "ElasticNet": "Elastic-net",
                    **{m: m for m in models if m != "ElasticNet"}}
    short = ["EN", "SVM", "RF", "GBM", "XGB", "CB", "LGBM", "TabPFN"]
    rank_colors = ["#123D68", "#326FA4", "#86B3D6", "#D9E2EB"]
    cmap = LinearSegmentedColormap.from_list(
        "correlation_blue", ["#E9EFF5", "#87B4D6", "#286294", "#123D68"])
    fig = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    fig.text(0.04, 0.955, "A", fontsize=12, weight="bold")
    ax = fig.add_axes([0.31, 0.505, 0.665, 0.420])
    for row, feature in enumerate(top):
        ax.axhline(row, color="#EDF0F3", linewidth=0.55, zorder=0)
        for col, model in enumerate(models):
            rank = float(ranks.loc[feature, model])
            band = 0 if rank <= 3 else 1 if rank <= 6 else 2 if rank <= 12 else 3
            ax.scatter(col, row, s=215, color=rank_colors[band], edgecolor="#CBD5DF", lw=0.25)
            txt = str(int(rank)) if rank.is_integer() else f"{rank:.1f}"
            ax.text(col, row, txt, ha="center", va="center", fontsize=7 if len(txt) < 4 else 6,
                    color="white" if band <= 1 else "#24384B", weight="bold" if rank == 1 else "normal")
    ax.set(xlim=(-0.5, 7.5), ylim=(11.55, -0.55))
    ax.set_yticks(range(12), [labels.get(f, display_name(f)) for f in top], fontsize=8)
    ax.set_xticks(range(8), [model_labels[m] for m in models], rotation=35, ha="right", fontsize=7.5)
    ax.tick_params(axis="both", length=0, pad=8); ax.spines[:].set_visible(False)
    for tick in ax.get_xticklabels():
        if tick.get_text() == "TabPFN": tick.set_weight("bold")
    legend = [Line2D([], [], ls="", marker="o", markersize=7, color=c, label=t)
              for c, t in zip(rank_colors, ["Ranks 1–3", "4–6", "7–12", "13–36"])]
    fig.legend(handles=legend, loc="center", bbox_to_anchor=(0.635, 0.405), ncol=4,
               frameon=False, fontsize=7.5)
    fig.text(0.04, 0.350, "B", fontsize=12, weight="bold")
    bx = fig.add_axes([0.22, 0.040, 0.28, 0.260])
    for i, model in enumerate(all_models):
        value = float(importance.loc["EDU", model] * 100)
        color = "#B65D16" if model == "TabPFN" else "#326FA4"
        bx.hlines(i, 0, value, color=color, linewidth=1.6)
        bx.scatter(value, i, color=color, s=28 if model == "TabPFN" else 22,
                   marker="D" if model == "TabPFN" else "o")
        bx.text(value + 3.5, i, f"{value:.1f}", va="center", fontsize=7.5,
                weight="bold" if model == "TabPFN" else "normal")
    bx.set(xlim=(0, 100), ylim=(8.55, -0.55), xlabel="Normalized importance (%)")
    bx.set_xticks([0, 25, 50, 75, 100])
    bx.set_yticks(range(9), [model_labels[m] for m in all_models], fontsize=7)
    bx.tick_params(axis="y", length=0, pad=5); bx.spines[["top", "right", "left"]].set_visible(False)
    for tick in bx.get_yticklabels():
        if tick.get_text() == "TabPFN": tick.set_weight("bold")
    fig.text(0.575, 0.350, "C", fontsize=12, weight="bold")
    cx = fig.add_axes([0.645, 0.040, 0.315, 0.260])
    for row in range(8):
        for col in range(8):
            rho = float(subcorr.iloc[row, col])
            if row == col:
                cx.text(col, row, "—", ha="center", va="center", fontsize=7, color="#909AA5")
            else:
                cx.scatter(col, row, s=230 * rho, color=cmap(rho), lw=0)
                cx.text(col, row, f"{rho:.2f}".removeprefix("0"), ha="center", va="center",
                        fontsize=6.1, color="white" if rho >= 0.56 else "#18354E")
    cx.set(xlim=(-0.5, 7.5), ylim=(7.5, -0.5), aspect="equal")
    cx.set_xticks(range(8), short, rotation=45, ha="right", fontsize=6.5)
    cx.set_yticks(range(8), short, fontsize=6.5)
    cx.tick_params(axis="both", length=0, pad=4); cx.spines[:].set_visible(False)
    save_figure(fig, MAIN_OUT, "Figure6_Predictor_importance_and_rank_agreement", dpi=600)


def copy_existing(stem_source, stem_target):
    for suffix in (".png", ".pdf"):
        source = stem_source.with_suffix(suffix)
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, SUPP_OUT / f"{stem_target}{suffix}")


def supplementary_s1():
    copy_existing(FIG_SUPP / "FigureS1_missingness", "Supplementary_Figure_S1_Missingness")


AXIS_LABEL = {
    "cat__EDU": "Education  (0 = academic, 1 = illiterate/primary, 2 = secondary)",
    "num__age": "Age (standardized)",
    "cat__sleep_quality": "Sleep quality  (0 = good, 1 = poor)",
    "cat__iadl": "IADL  (0 = dependent, 1 = independent)",
    "num__visual_acuity_both": "Binocular visual acuity (standardized)",
}


def supplementary_s3_dependence():
    top_idx = np.argsort(np.mean(np.abs(np.asarray(explanation.values)), axis=0))[::-1][:5]
    pretty_names = [display_name(x) for x in transformed_names]
    fig = plt.figure(figsize=(13.6, 14.3))
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.28)
    for rank, idx in enumerate(top_idx):
        ax = fig.add_subplot(gs[rank // 2, rank % 2])
        shap.dependence_plot(
            int(idx), np.asarray(explanation.values), X_shap,
            feature_names=pretty_names, interaction_index="auto", ax=ax, show=False,
        )
        raw = transformed_names[idx]
        ax.set_xlabel(AXIS_LABEL.get(raw, display_name(raw)))
        ax.set_ylabel(f"SHAP value for {display_name(raw)}")
        ax.set_title(f"{chr(65 + rank)}  {display_name(raw)}", loc="left", fontweight="bold")
    empty = fig.add_subplot(gs[2, 1]); empty.axis("off")
    save_figure(fig, SUPP_OUT, "Supplementary_Figure_S3_SHAP_dependence", dpi=300)


def crop_title(img, fraction=0.045):
    out = img.convert("RGB").copy()
    ImageDraw.Draw(out).rectangle([0, 0, out.width, int(out.height * fraction)], fill="white")
    return out


def supplementary_s4_cases():
    manifest = pd.read_csv(TABLES / "SHAP_local_case_manifest.csv").set_index("case_type")
    order = ["TP", "TN", "FP", "FN"]
    labels = {"TP": "True positive", "TN": "True negative",
              "FP": "False positive", "FN": "False negative"}
    width, height, margin, gap = 4290, 4730, 100, 90
    cell_w = (width - 2 * margin - gap) // 2
    cell_h = (height - 2 * margin - gap) // 2
    canvas = Image.new("RGB", (width, height), "white")
    for i, case in enumerate(order):
        source = FIG_SUPP / f"FigureS7_SHAP_waterfall_{case}.png"
        img = crop_title(Image.open(source), 0.040)
        r, c = divmod(i, 2)
        box = (margin + c * (cell_w + gap), margin + r * (cell_h + gap), cell_w, cell_h)
        p = float(manifest.loc[case, "predicted_probability"])
        paste_panel(canvas, img, box, chr(65 + i), f"{labels[case]} — predicted risk {p:.3f}",
                    letter_size=52, title_size=40)
    save_pil(canvas, SUPP_OUT, "Supplementary_Figure_S4_SHAP_local_cases", dpi=300)


def create_manifest_and_zip():
    records = []
    for group, folder in (("main", MAIN_OUT), ("supplementary", SUPP_OUT)):
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() in {".png", ".pdf"}:
                records.append({"group": group, "file": path.name, "bytes": path.stat().st_size})
    pd.DataFrame(records).to_csv(OUT / "figure_manifest.csv", index=False)
    summary = {
        "main_figure_count": len(list(MAIN_OUT.glob("*.png"))),
        "supplementary_figure_count": len(list(SUPP_OUT.glob("*.png"))),
        "png_count": len(list(OUT.rglob("*.png"))),
        "pdf_count": len(list(OUT.rglob("*.pdf"))),
        "model_refits_in_this_script": 0,
        "source": "locked outputs and in-memory objects from the completed Moshabeh notebook",
    }
    (OUT / "README.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    zip_path = ROOT / "Moshabeh_all_manuscript_figures.zip"
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file(): zf.write(path, path.relative_to(OUT.parent))
        script_path = ROOT / "generate_all_submission_figures.py"
        if script_path.exists(): zf.write(script_path, script_path.name)
    print(json.dumps(summary, indent=2))
    print("ZIP ready:", zip_path)


figure2_and_s2()
figure3_hexbin()
figure4_calibration()
figure5_shap_subgroups()
figure6_importance()
supplementary_s1()
supplementary_s3_dependence()
supplementary_s4_cases()
create_manifest_and_zip()
