"""
Create bootstrap estimates for given results and perform statistical significance testing.
"""

# STD
import argparse
import os
from collections import defaultdict
from typing import List

# EXT
from deepsig import multi_aso
import dill
import numpy as np
from src.eval import evaluate_confidences, get_target_function
from tqdm import tqdm

# CONST
METRICS = ["brier_score", "ece", "smece", "auroc"]


def compute_confidence_intervals_and_test_significance(
    result_dirs: List[str],
    num_bootstrap_samples: int,
    decision_threshold: float = 0.35,
    confidence_level: float = 0.95,
):
    """
    Compute confidence intervals through a bootstrapping estimator and compute significance using the ASO test.
    Lastly, print the results in a Latex-friendly formatting.

    Parameters
    ----------
    result_dirs: List[str]
        List of directories in which to look for result dill files.
    num_bootstrap_samples: int
        Number of bootstrap samples used to compute confidence intervals.
    decision_threshold: float
        Decision threshold for significance testing. Default is 0.35.
    confidence_level: float
        Confidence level used for significance level. Default is 0.95.
    """
    # Dictionary mapping from metric to method and its bootstrap samples
    orig_results = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    bootstrap_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_methods = []

    for result_dir in result_dirs:
        for result_path in os.listdir(result_dir):
            if not result_path.endswith(".dill"):
                continue

            with open(os.path.join(result_dir, result_path), "rb") as result_file:
                results = dill.load(result_file)

            # Determine the main of current method
            if "baseline_name" in results["info"]:
                method = results["info"]["baseline_name"]

            else:
                method = "auxiliary"

                if results["info"]["use_binary_targets"]:
                    method += "_binary"

                else:
                    method += "_clustering"

                if results["info"]["num_steps_temperature_scaling"] > 0:
                    method += "_temp_scaling"

                if results["info"]["use_isotonic_regression"]:
                    method += "_isotonic_regression"

                method += "_" + "_".join(results["info"]["input_parts"])

            all_methods.append(method)

            for split_name, split_data in results["eval_data"].items():
                all_confidences = np.array(split_data["all_confidences"])
                all_correctness = np.array(split_data["all_correctness"])

                if "all_targets" not in split_data:
                    target_function = get_target_function(
                        all_confidences, all_correctness
                    )
                    all_targets = target_function(all_confidences)
                else:
                    all_targets = np.array(split_data["all_targets"])

                num_points = len(all_targets)

                # Add original results
                for metric, res in evaluate_confidences(
                    split_name=split_name,
                    all_confidences=list(all_confidences),
                    all_targets=all_targets,
                    all_correctness=list(all_correctness),
                ).items():
                    orig_results[split_name][metric][method] = res

                # Create bootstrap samples by sampling indices (with replacement)
                for _ in tqdm(range(num_bootstrap_samples)):
                    # Make sure to use the same indices here, otherwise we would compare measurements from different
                    # datapoints.
                    indices = np.random.choice(range(num_points), size=num_points)
                    bootstrap_confidences = list(all_confidences[indices])
                    bootstrap_correctness = list(all_correctness[indices])
                    bootstrap_targets = list(all_targets[indices])

                    for metric, res in evaluate_confidences(
                        split_name=split_name,
                        all_confidences=bootstrap_confidences,
                        all_targets=bootstrap_targets,
                        all_correctness=bootstrap_correctness,
                    ).items():
                        bootstrap_results[split_name][metric][method].append(res)

    # Perform significance testing
    # Mapping split -> metric -> method -> bool
    is_significant = defaultdict(lambda: defaultdict(lambda: dict()))
    for split_name, split_data in bootstrap_results.items():
        for metric, metric_results in split_data.items():
            metric_results = dict(metric_results)

            # Make sure that higher = better
            if "brier_score" in metric or "ece" in metric:
                metric_results = {
                    method: 1 - np.array(method_scores)
                    for method, method_scores in metric_results.items()
                }

            eps_min = multi_aso(
                dict(metric_results),
                confidence_level=confidence_level,
                num_bootstrap_iterations=100,
            )

            for i, method in enumerate(metric_results.keys()):
                row = eps_min[i, :]
                row = np.delete(row, i)  # Delete the comparison of a method with itself

                is_significant[split_name][metric][method] = np.all(
                    row < decision_threshold
                )

    # Identify the best scores
    all_ranks = defaultdict(lambda: defaultdict(lambda: dict()))
    for split_name, split_results in orig_results.items():
        for metric, metric_results in split_results.items():
            metric_results = {
                method: 1 - np.array(method_scores)
                if "brier_score" in metric or "ece" in metric
                else np.array(method_scores)
                for method, method_scores in metric_results.items()
            }
            methods, ranks, scores = zip(
                *list(
                    sorted(
                        zip(
                            metric_results.keys(),
                            range(1, len(metric_results) + 1),
                            metric_results.values(),
                        ),
                        key=lambda tpl: tpl[2],
                    )
                )
            )

            # Identify maximum score(s)
            max_score = round(np.max(scores), 2)
            for method, score in zip(methods, scores):
                if np.round(score, 2) == max_score:
                    all_ranks[split_name][metric][method] = 1

                else:
                    all_ranks[split_name][metric][method] = 100

    # Compute and print results
    for split_name, split_data in bootstrap_results.items():
        print(f"##### {split_name} #####\n")

        for method in all_methods:
            method_str = f"{method} "

            for metric in METRICS:
                orig_val = f"{orig_results[split_name][f'{split_name}_{metric}'][method]:.2f}".lstrip(
                    "0"
                )

                rank = all_ranks[split_name][f"{split_name}_{metric}"][method]
                if rank == 1:
                    orig_val = "\mathbf{" + orig_val + "}"

                if is_significant[split_name][f"{split_name}_{metric}"][method]:
                    orig_val = "\\" + "underline{" + orig_val + "}"

                data = np.array(
                    bootstrap_results[split_name][f"{split_name}_{metric}"][method]
                )
                # Bootstrap estimator for standard deviation
                std = float(
                    np.sqrt(
                        np.sum((data - np.mean(data)) ** 2)
                        / (num_bootstrap_samples - 1)
                        + 1e-8
                    )
                )

                std_dev = f"{std:.2f}".lstrip("0")
                method_str += (
                    f"& ${orig_val}" + "{\scriptstyle\ \pm" + f"{std_dev}" + "}$"
                )

            print(method_str)

        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result-dirs", type=str, nargs="+", help="Dirs with _results.dill files."
    )

    parser.add_argument(
        "--num-bootstrap-samples",
        type=int,
        nargs="+",
        help="Paths to result files.",
        default=100,
    )

    args = parser.parse_args()

    compute_confidence_intervals_and_test_significance(
        result_dirs=args.result_dirs, num_bootstrap_samples=args.num_bootstrap_samples
    )
