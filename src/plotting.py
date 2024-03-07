"""
Plotting-related code.
"""

# STD
from typing import Optional, Dict, List

# EXT
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_reliability_diagram(
    all_confidences: List[float],
    all_correctness: List[int],
    num_bins: int = 10,
    save_path: Optional[str] = None,
    display_percentages: bool = True,
    success_percentage: float = 1,
):
    """
    Create a reliability diagram illustrating the calibration of a model.

    Parameters
    ----------
    all_confidences: List[float]
        List of all the confidence scores ona given split.
    all_correctness: List[int]
        List of all the results of answers of the target LLM on a split, as zeros and ones for correctness.
    num_bins: int
        Number of bins used for plotting. Defaults to 10.
    save_path: Optional[str]
        Path to save the plot to. If None, the plot is shown directly.
    display_percentages: bool
        Indicate whether bins should include the percentage of points falling into them. Defaults to True.
    success_percentage: float
        The percentage of successful confidence scores. Defaults to 1, but if is not will be printed in a box on the
        bottom left.
    """
    bins = np.arange(0.0, 1.0, 1.0 / num_bins)
    bins_per_prediction = np.digitize(all_confidences, bins)
    df = pd.DataFrame(
        {
            "y_pred": all_confidences,
            "y": all_correctness,
            "pred_bins": bins_per_prediction,
        }
    )

    grouped_by_bins = df.groupby("pred_bins")
    # calculate the mean y and predicted probabilities per bin
    grouped_bins = grouped_by_bins.mean()
    grouped_bins = grouped_bins["y"].reindex(range(1, num_bins + 1), fill_value=0)
    bin_values = grouped_bins.values

    # calculate the number of items per bin
    bin_sizes = grouped_by_bins["y"].count()
    bin_sizes = bin_sizes.reindex(range(1, num_bins + 1), fill_value=0)

    plt.figure(figsize=(4, 4), dpi=200)
    ax = plt.gca()
    ax.grid(visible=True, axis="both", which="major", linestyle=":", color="grey")
    step_size = 1.0 / num_bins

    # Get bar colors
    bar_colors = []

    # Display the amount of points that fall into each bin via different shading
    if display_percentages:
        total = sum(bin_sizes.values)

        for i, (bin, bin_size) in enumerate(zip(bins, bin_sizes.values)):
            bin_percentage = bin_size / total * success_percentage
            cmap = matplotlib.cm.get_cmap("Blues")
            bar_colors.append(cmap(min(0.9999, bin_percentage + 0.2)))

    plt.bar(
        bins + step_size / 2,
        bin_values,
        width=0.09,
        alpha=0.8,
        color=bar_colors,  # "royalblue",
        edgecolor="black",
    )
    plt.plot(
        np.arange(0, 1 + 0.05, 0.05),
        np.arange(0, 1 + 0.05, 0.05),
        color="black",
        alpha=0.4,
        linestyle="--",
    )

    # Now add the percentage value of points per bin as text
    if display_percentages:
        total = sum(bin_sizes.values)
        eps = 0.01

        for i, (bin, bin_size) in enumerate(zip(bins, bin_sizes.values)):
            bin_percentage = round(bin_size / total * success_percentage * 100, 2)

            # Omit labelling for very small bars
            if bin_size == 0 or bin_values[i] < 0.2:
                continue

            plt.annotate(
                f"{bin_percentage} %",
                xy=(bin + step_size / 2, bin_values[i] - eps),
                ha="center",
                va="top",
                rotation=90,
                color="white" if bin_percentage > 40 else "black",
                alpha=0.7 if bin_percentage > 40 else 0.8,
                fontsize=10,
            )

        # Display success percentage if it is not 1
        if success_percentage < 1:
            plt.annotate(
                f"Success: {round(success_percentage * 100, 2)} %",
                xy=(-0.19, -0.135),
                color="royalblue",
                fontsize=10,
                alpha=0.7,
                annotation_clip=False,
                bbox=dict(facecolor="none", edgecolor="royalblue", pad=4.0, alpha=0.7),
            )

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Confidence", fontsize=14, alpha=0.8)
    plt.ylabel("Accuracy", fontsize=14, alpha=0.8)
    plt.tight_layout()

    if save_path is None:
        plt.show()

    else:
        plt.savefig(save_path)
