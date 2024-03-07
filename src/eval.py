"""
Module implementing evaluation functions.
"""

# STD
import re
from typing import Dict, Optional, List, Union, Tuple, Callable

# EXT
import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from relplot.metrics import smECE_slow as smece
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression

# PROJECT

# GLOBALS
ROUGE = evaluate.load("rouge")  # Preload here to avoid re-loading on function call


def evaluate_model(
    calibration_model: AutoModelForSequenceClassification,
    dataloaders: Dict[str, DataLoader],
    calibration_targets: Dict[str, float],
    device: torch.device | str,
    use_binary_targets: bool,
    return_eval_data: bool = True,
) -> Tuple[Dict[str, float], Union[Dict[str, float]] | Dict[str, List[int | float]]]:
    """
    Evaluate an external calibration model.

    Parameters
    ----------
    calibration_model: AutoModelForSequenceClassification
        Calibration model to be evaluated.
    dataloaders: Dict[str, DataLoader]
        Dictionary mapping from split name to dataloader to evaluate on.
    calibration_targets: Dict[str, float]
        Dictionary mapping from question id to calibration targets.
    device: torch.device | str
        Device the calibration model lives on.
    use_binary_targets: bool
        Whether the calibration model uses binary targets instead of clustering-based targets.
    return_eval_data: bool
        Flag indicating whether we should also return all the data evaluations are based on (useful for bootstrapping
        analysis). Defaults to True.

    Returns
    -------
    Dict[str, float] | Union[Dict[str, float], Dict[str, List[int | float]]]
        Returns either only a dictionary mapping from a metric name to its value, or the same dictionary alongside the
        raw data that metrics are based on, in a dictionary mapping from the data name to lists of floats or ints.

    """
    eval_data = {}
    metrics = {}

    with torch.no_grad():
        for split_name, test_dataloader in dataloaders.items():
            all_correctness = []
            all_confidences = []
            all_targets = []

            if "test" not in split_name:
                continue

            for batch in test_dataloader:
                input_ids = batch["input_ids"].squeeze(1).to(device)
                attention_mask = batch["attention_mask"].squeeze(1).to(device)

                if use_binary_targets:
                    all_targets += batch["correctness"].tolist()
                    outputs = calibration_model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    preds = F.softmax(outputs.logits, dim=-1)[:, 1]

                else:
                    all_targets += [
                        calibration_targets[question_id]
                        for question_id in batch["question_id"]
                    ]
                    outputs = calibration_model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    preds = F.softmax(outputs.logits, dim=-1)[:, 1]

                preds = preds.cpu().tolist()
                all_confidences += preds
                all_correctness += batch["correctness"].tolist()

            metrics.update(
                {
                    f"{split_name}_ece": ece(
                        y_true=all_targets, y_pred=all_confidences
                    ),
                    f"{split_name}_smece": smece(
                        f=np.array(all_confidences), y=np.array(all_targets)
                    ),
                    f"{split_name}_bier_score": brier_score_loss(
                        y_true=all_correctness, y_prob=all_confidences
                    ),
                    f"{split_name}_accuracy": np.mean(all_correctness),
                    f"{split_name}_auroc": roc_auc_score(
                        y_true=all_correctness, y_score=all_confidences
                    ),
                }
            )

            eval_data[split_name] = {
                "all_correctness": all_correctness,
                "all_confidences": all_confidences,
                "all_targets": all_targets,
            }

    if not return_eval_data:
        return metrics

    else:
        return metrics, eval_data


def evaluate_confidences(
    split_name: str,
    all_confidences: List[float],
    all_correctness: List[int],
    all_targets: Optional[List[float]] = None,
    num_bins: int = 10,
    add_name: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate the confidence scores produced

    Parameters
    ----------
    split_name: str
        Name of the current split.
    all_confidences: List[float]
        All confidences values for a given split.
    all_correctness: List[int]
        All information relating to the correctness of the target LLM's answer on a given split.
    all_targets: Optional[List[float]]
        All the desired calibration targets. If None, targets will computed via binning.
    num_bins: int
        Number of bins used for the target function. Default is 10.
    add_name: Optional[str]
        An additional run name to add to results. Default is None.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping from metric names to values.

    """
    # Compute targets
    if all_targets is None:
        target_func = get_target_function(all_confidences, all_correctness, num_bins)
        all_targets = target_func(all_confidences)

    infix = ""
    if add_name is not None:
        infix = f"{add_name}_"

    metrics = {
        f"{split_name}_{infix}ece": ece(y_true=all_targets, y_pred=all_confidences),
        f"{split_name}_{infix}smece": smece(
            f=np.array(all_confidences), y=np.array(all_targets)
        ),
        f"{split_name}_{infix}brier_score": brier_score_loss(
            y_true=all_correctness, y_prob=all_confidences
        ),
        f"{split_name}_{infix}auroc": roc_auc_score(
            y_true=all_correctness, y_score=all_confidences
        ),
    }

    return metrics


def check_answer_correctness(
    correct_answers: List[str],
    model_answers: List[str],
    rouge_threshold: float = 0.3,
) -> List[bool]:
    """
    Check whether a given answer is correct. This uses the heuristic by Kuhn et al. (2023), checking whether the ROUGE-L
    score is higher than some threshold.
    Additionally, we check via simply string matching whether the correct answer is included in the model answer.

    Parameters
    ----------
    correct_answers: List[str]
        Reference answers.
    model_answers: List[str]
        Model generations to compare to the reference answer.
    rouge_threshold: float
        Threshold of ROUGE-L scores over which an answer is deemed correct. Default is 0.3.

    Returns
    -------
    List[bool]
        Whether the given answer was deemed correct.
    """
    global ROUGE

    results = [
        # Add the second criterion to accommodate CoT answers that might be longer and therefore obtain lower ROUGE
        # scores but are still correct.
        res >= rouge_threshold or correct_answer in model_answer
        for res, correct_answer, model_answer in zip(
            ROUGE.compute(
                predictions=model_answers,
                references=correct_answers,
                use_aggregator=False,
            )["rougeL"],
            correct_answers,
            model_answers,
        )
    ]

    return results


def ece(y_true: np.array, y_pred: np.array, n_bins: int = 10) -> float:
    """
    Calculate the Expected Calibration Error: for each bin, the absolute difference between
    the mean fraction of positives and the average predicted probability is taken. The ECE is
    the weighed mean of these differences.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels.
    y_pred: np.ndarray
        The predicted probabilities
    n_bins: int
        The number of bins to use.
    Returns
    -------
    ece: float
        The expected calibration error.
    """
    n = len(y_pred)
    bins = np.arange(0.0, 1.0, 1.0 / n_bins)
    bins_per_prediction = np.digitize(y_pred, bins)

    df = pd.DataFrame({"y_pred": y_pred, "y": y_true, "pred_bins": bins_per_prediction})

    grouped_by_bins = df.groupby("pred_bins")
    # calculate the mean y and predicted probabilities per bin
    binned = grouped_by_bins.mean()

    # calculate the number of items per bin
    binned_counts = grouped_by_bins["y"].count()

    # calculate the proportion of data per bin
    binned["weight"] = binned_counts / n

    weighed_diff = abs(binned["y_pred"] - binned["y"]) * binned["weight"]
    return weighed_diff.sum()


def extract_verbalized_confidence(
    expressions: List[str],
    mode: str = "quantitative",
    expression_mapping: Optional[Dict[str, float]] = None,
) -> Tuple[List[float], List[bool]]:
    """
    Extract the confidence scores from the verbalized confidence generated from a model.

    Parameters
    ----------
    expressions: List[str]
        List of expressions containing verbalized confidence.
    mode: str
        Whether the confidence is "qualitative" or "quantitative". Defaults to the latter.
    expression_mapping: Optional[Dict[str, float]]
        If the mode is "qualitative", supply a dictionary that maps from confidence expression to numerical values.

    Returns
    -------
    Tuple[List[float], List[bool]]
        Extracted confidence scores, as well as list of boolean values indicating whether the extraction was successful.
    """
    assert mode in (
        "qualitative",
        "quantitative",
    ), f"Mode has to be either qualitative or quantitative, but {mode} found."

    if mode == "qualitative":
        assert (
            expression_mapping is not None
        ), "'expression_mapping' has to be specified for qualitative mode."

    confidences, successful = [], []

    for expression in expressions:
        if mode == "qualitative":
            template = rf"({'|'.join(expression_mapping.keys())})"

        else:
            # With the template below, try to capture anything like: 95%, 95 %, 96.666, 100, etc.
            template = r"\d{1,3}(?:\.\d+)?\s?\%?"

        try:
            res = re.search(template, expression).group(0)

            if mode == "qualitative":
                conf = expression_mapping[res]

            else:
                conf = float(res.replace("%", "")) / 100

                if not (0 <= conf <= 1):
                    successful.append(False)
                    continue

            successful.append(True)
            confidences.append(conf)

        except AttributeError:
            successful.append(False)

    return confidences, successful


def get_target_function(
    all_confidences: List[float], all_correctness: List[int], num_bins: int = 10
) -> Callable:
    """
    Create a function that maps confidences to their target values, based on binning in an ECE-style manner.

    Parameters
    ----------
    all_confidences: List[float]
        Raw, uncalibrated confidences.
    all_correctness: List[int]
        List of whether the target model was correct as ones and zeros.
    num_bins: int
        Number of bins to consider.

    Returns
    -------
    Callable
        Function mapping a confidence to its corresponding target.
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
    targets = grouped_by_bins.mean()["y"].values

    return np.vectorize(lambda conf: targets[np.abs(conf - targets).argmin()])
