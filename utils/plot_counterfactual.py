import os
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec


def plot_counterfactual(
    x: np.ndarray,  # (L,F) original
    x_cf: np.ndarray,  # (L,F) counterfactual
    *,
    feature_names: Optional[List[str]] = None,
    edit_segment: Optional[Tuple[int, int]] = None,
    show_diff: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    score_original: Optional[float] = None,
    score_cf: Optional[float] = None,
    threshold: Optional[float] = None,
    show_heatmap: bool = False,
    max_features_per_col: int = 6,
    style: str = "modern",  # "modern", "paper", "minimal"
):
    """
    Enhanced counterfactual plot for time-series windows with multiple layout options.

    Args:
        x: original window (L,F)
        x_cf: counterfactual window (L,F)
        feature_names: optional list of feature names
        edit_segment: (start, end) tuple if segment substitution was used
        show_diff: whether to show |x - x_cf|
        title: plot title (will be enhanced with metrics)
        save_path: path to save figure
        score_original: reconstruction score of original
        score_cf: reconstruction score of counterfactual
        threshold: anomaly threshold
        show_heatmap: add heatmap visualization of differences
        max_features_per_col: max features per column in multi-column layout
        style: visual style preset
    """
    assert x.shape == x_cf.shape, "x and x_cf must have same shape"
    L, F = x.shape

    if feature_names is None:
        feature_names = [f"Feature {i + 1}" for i in range(F)]

    # Apply style presets
    if style == "modern":
        plt.style.use("seaborn-v0_8-darkgrid")
        color_orig = "#2E86AB"  # Blue
        color_cf = "#A23B72"  # Purple
        color_diff = "#F18F01"  # Orange
        color_edit = "#C73E1D"  # Red
    elif style == "paper":
        plt.style.use("seaborn-v0_8-paper")
        color_orig = "#1f77b4"
        color_cf = "#d62728"
        color_diff = "#ff7f0e"
        color_edit = "#bcbd22"
    else:  # minimal
        plt.style.use("default")
        color_orig = "#3498db"
        color_cf = "#e74c3c"
        color_diff = "#95a5a6"
        color_edit = "#f39c12"

    # Calculate metrics
    total_change = np.abs(x - x_cf).sum()
    mean_change = np.abs(x - x_cf).mean()
    max_change = np.abs(x - x_cf).max()
    n_changed_features = (np.abs(x - x_cf).sum(axis=0) > 1e-6).sum()

    # Find features with largest changes
    feature_changes = np.abs(x - x_cf).sum(axis=0)
    top_changed_indices = np.argsort(feature_changes)[-3:][::-1]

    # Build enhanced title
    title_parts = []
    if title:
        title_parts.append(title)

    if score_original is not None and score_cf is not None:
        score_reduction = ((score_original - score_cf) / score_original) * 100
        title_parts.append(
            f"Score: {score_original:.4f} → {score_cf:.4f} ({score_reduction:+.1f}%)"
        )

    title_parts.append(
        f"Changed: {n_changed_features}/{F} features | "
        f"Total Δ: {total_change:.2f} | Mean Δ: {mean_change:.4f}"
    )

    enhanced_title = "\n".join(title_parts)

    # Determine layout
    if F <= max_features_per_col and not show_heatmap:
        # Single column layout
        n_rows = F if not show_diff else 2 * F
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(14, min(2.5 * n_rows, 30)),
            sharex=True,
        )
        if n_rows == 1:
            axes = [axes]

        _plot_features_single_column(
            axes,
            x,
            x_cf,
            feature_names,
            edit_segment,
            show_diff,
            color_orig,
            color_cf,
            color_diff,
            color_edit,
        )

    else:
        # Multi-column or heatmap layout
        fig = plt.figure(figsize=(18, min(3 * F + 4, 30)))

        if show_heatmap:
            # Layout with heatmap
            gs = GridSpec(
                F + 2,
                3,
                figure=fig,
                hspace=0.4,
                wspace=0.3,
                height_ratios=[1] * F + [0.5, 0.5],
            )

            # Time series plots (left 2 columns)
            axes_ts = []
            for i in range(F):
                ax = fig.add_subplot(gs[i, :2])
                axes_ts.append(ax)

            _plot_features_single_column(
                axes_ts,
                x,
                x_cf,
                feature_names,
                edit_segment,
                show_diff=False,
                color_orig=color_orig,
                color_cf=color_cf,
                color_diff=color_diff,
                color_edit=color_edit,
            )

            # Heatmap (right column)
            ax_heatmap = fig.add_subplot(gs[:F, 2])
            _plot_difference_heatmap(ax_heatmap, x, x_cf, feature_names, edit_segment)

            # Summary statistics (bottom)
            ax_summary = fig.add_subplot(gs[F:, :])
            _plot_summary_statistics(
                ax_summary,
                x,
                x_cf,
                feature_names,
                feature_changes,
                score_original,
                score_cf,
                threshold,
            )

        else:
            # Two-column layout for many features
            n_cols = 2
            n_rows = int(np.ceil(F / n_cols))
            gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

            for idx, f in enumerate(F):
                row = idx // n_cols
                col = idx % n_cols
                ax = fig.add_subplot(gs[row, col])

                ax.plot(
                    x[:, f], label="Original", linewidth=2, color=color_orig, alpha=0.8
                )
                ax.plot(
                    x_cf[:, f],
                    label="Counterfactual",
                    linewidth=2,
                    color=color_cf,
                    alpha=0.8,
                    linestyle="--",
                )

                if edit_segment is not None:
                    ax.axvspan(
                        edit_segment[0], edit_segment[1], color=color_edit, alpha=0.15
                    )

                # Highlight if significant change
                if f in top_changed_indices:
                    ax.set_facecolor("#fffacd")
                    ax.set_title(
                        f"{feature_names[f]} ⚠️", fontweight="bold", fontsize=10
                    )
                else:
                    ax.set_title(feature_names[f], fontsize=10)

                ax.grid(True, alpha=0.3, linestyle=":")
                ax.set_ylabel("Value", fontsize=8)

                if idx == 0:
                    ax.legend(loc="upper right", fontsize=8)

                if row == n_rows - 1:
                    ax.set_xlabel("Time Step", fontsize=8)

    # Add overall title
    fig.suptitle(enhanced_title, fontsize=12, weight="bold", y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()


def _plot_features_single_column(
    axes,
    x,
    x_cf,
    feature_names,
    edit_segment,
    show_diff,
    color_orig,
    color_cf,
    color_diff,
    color_edit,
):
    """Helper to plot features in single column layout"""
    L, F = x.shape
    row = 0

    for f in range(F):
        ax = axes[row]

        # Main comparison plot
        ax.plot(
            x[:, f],
            label="Original",
            linewidth=2.5,
            color=color_orig,
            alpha=0.85,
            marker="o",
            markersize=2,
            markevery=max(1, L // 20),
        )
        ax.plot(
            x_cf[:, f],
            label="Counterfactual",
            linewidth=2.5,
            color=color_cf,
            alpha=0.85,
            linestyle="--",
            marker="s",
            markersize=2,
            markevery=max(1, L // 20),
        )

        if edit_segment is not None:
            ax.axvspan(
                edit_segment[0],
                edit_segment[1],
                color=color_edit,
                alpha=0.2,
                label="Edit Region" if f == 0 else None,
            )

        # Calculate change magnitude
        change = np.abs(x[:, f] - x_cf[:, f]).sum()
        ax.set_ylabel(
            f"{feature_names[f]}\n(Δ={change:.2f})", fontsize=10, weight="bold"
        )
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if f == 0:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

        row += 1

        # Difference subplot
        if show_diff:
            ax_diff = axes[row]
            diff = np.abs(x[:, f] - x_cf[:, f])
            ax_diff.fill_between(range(L), 0, diff, color=color_diff, alpha=0.6)
            ax_diff.plot(diff, color=color_diff, linewidth=1.5)

            # Mark significant changes
            threshold_diff = diff.mean() + 2 * diff.std()
            significant = diff > threshold_diff
            if significant.any():
                ax_diff.scatter(
                    np.where(significant)[0],
                    diff[significant],
                    color="red",
                    s=30,
                    zorder=5,
                    marker="^",
                    label="Significant",
                )

            ax_diff.set_ylabel(f"|Δ|", fontsize=9)
            ax_diff.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
            ax_diff.spines["top"].set_visible(False)
            ax_diff.spines["right"].set_visible(False)
            row += 1

    axes[-1].set_xlabel("Time Step", fontsize=11, weight="bold")


def _plot_difference_heatmap(ax, x, x_cf, feature_names, edit_segment):
    """Plot heatmap of differences between original and counterfactual"""
    diff = np.abs(x - x_cf).T  # (F, L)

    im = ax.imshow(diff, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Difference|", rotation=270, labelpad=15, fontsize=9)

    # Set ticks
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel("Time Step", fontsize=9)
    ax.set_title("Difference Heatmap", fontsize=10, weight="bold")

    # Highlight edit segment
    if edit_segment is not None:
        ax.axvline(
            edit_segment[0], color="blue", linewidth=2, linestyle="--", alpha=0.7
        )
        ax.axvline(
            edit_segment[1], color="blue", linewidth=2, linestyle="--", alpha=0.7
        )

    ax.grid(False)


def _plot_summary_statistics(
    ax, x, x_cf, feature_names, feature_changes, score_original, score_cf, threshold
):
    """Plot summary statistics and bar chart of feature changes"""
    ax.axis("off")

    # Bar chart of top changed features
    top_n = min(10, len(feature_names))
    top_indices = np.argsort(feature_changes)[-top_n:][::-1]

    ax_bar = plt.subplot(111)
    bars = ax_bar.barh(
        range(top_n),
        feature_changes[top_indices],
        color=plt.cm.RdYlGn_r(feature_changes[top_indices] / feature_changes.max()),
    )

    ax_bar.set_yticks(range(top_n))
    ax_bar.set_yticklabels([feature_names[i] for i in top_indices], fontsize=9)
    ax_bar.set_xlabel("Total Change", fontsize=10, weight="bold")
    ax_bar.set_title("Top Changed Features", fontsize=11, weight="bold")
    ax_bar.grid(axis="x", alpha=0.3, linestyle=":")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax_bar.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            ha="left",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )


def plot_counterfactual_comparison(
    originals: List[np.ndarray],
    counterfactuals: List[np.ndarray],
    feature_names: Optional[List[str]] = None,
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    scores_original: Optional[List[float]] = None,
    scores_cf: Optional[List[float]] = None,
):
    """
    Plot multiple counterfactual examples side-by-side for comparison.

    Args:
        originals: List of original windows
        counterfactuals: List of counterfactual windows
        feature_names: Feature names
        titles: Title for each example
        save_path: Path to save
        scores_original: Original scores
        scores_cf: Counterfactual scores
    """
    n_examples = len(originals)
    L, F = originals[0].shape

    if feature_names is None:
        feature_names = [f"F{i}" for i in range(F)]

    # Select top 5 most changed features overall
    all_changes = []
    for x, x_cf in zip(originals, counterfactuals):
        all_changes.append(np.abs(x - x_cf).sum(axis=0))

    avg_changes = np.mean(all_changes, axis=0)
    top_features = np.argsort(avg_changes)[-5:][::-1]

    # Create grid
    fig, axes = plt.subplots(
        len(top_features),
        n_examples,
        figsize=(5 * n_examples, 3 * len(top_features)),
        sharex=True,
        sharey="row",
    )

    if n_examples == 1:
        axes = axes.reshape(-1, 1)

    for col, (x, x_cf) in enumerate(zip(originals, counterfactuals)):
        for row, f in enumerate(top_features):
            ax = axes[row, col]

            ax.plot(x[:, f], label="Original", linewidth=2, alpha=0.7)
            ax.plot(x_cf[:, f], label="CF", linewidth=2, linestyle="--", alpha=0.7)

            if row == 0:
                title_text = titles[col] if titles else f"Example {col + 1}"
                if scores_original and scores_cf:
                    title_text += f"\n{scores_original[col]:.3f}→{scores_cf[col]:.3f}"
                ax.set_title(title_text, fontsize=10, weight="bold")

            if col == 0:
                ax.set_ylabel(feature_names[f], fontsize=9, weight="bold")

            if row == 0 and col == 0:
                ax.legend(fontsize=8)

            ax.grid(True, alpha=0.3, linestyle=":")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time Step", fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)
        print(f"✓ Saved comparison: {save_path}")
    else:
        plt.show()
