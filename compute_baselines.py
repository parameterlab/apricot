"""
Compute the following baselines for QA:
    - Sequence likelihood (w/ and w/o temperature scaling)
    - CoT Sequence likelihood (w/ and w/o temperature scaling)
    - Verbalized uncertainty (qualitative)
    - Verbalized uncertainty (quantitative)
"""

# STD
import argparse
from collections import defaultdict
from datetime import datetime
import os
from typing import List, Optional

# EXT
import dill
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from wandb.sdk.wandb_run import Run as WandBRun

# PROJECT
from src.constants import (
    MODEL_IDENTIFIER,
    DATASETS,
    DATA_DIR,
    NUM_IN_CONTEXT_SAMPLES,
    BASELINES_METHODS,
    PROJECT_NAME,
    DATASET_SPLIT_SIZES,
    PLATT_SCALING_BATCH_SIZE,
    PLATT_SCALING_LEARNING_RATE,
    PLATT_SCALING_NUM_STEPS,
    PLATT_SCALING_VALID_INTERVAL,
    QUALITATIVE_SCALE,
    EVAL_METRIC_ORDER,
)
from src.eval import (
    extract_verbalized_confidence,
    evaluate_confidences,
    get_target_function,
)
from src.plotting import plot_reliability_diagram
from src.utils import loop_dataloader


class PlattScaler(nn.Module):
    """
    Class that learns two scalers in order to transform LLM sequence likelihood into calibrated confidence scores.
    """

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, raw_probabilities: torch.FloatTensor) -> torch.FloatTensor:
        """
        Transform raw probabilities into calibrated probabilities through Platt scaling.

        Parameters
        ----------
        raw_probabilities: torch.FloatTensor
            Raw probabilities to be adjusted.

        Returns
        -------
        torch.FloatTensor
            Calibrated probabilities.
        """
        return F.sigmoid(raw_probabilities * self.scale + self.bias)

    def train_scaler(
        self,
        train_probabilities: torch.FloatTensor,
        train_targets: torch.FloatTensor,
        valid_probabilities: torch.FloatTensor,
        valid_targets: torch.FloatTensor,
        batch_size: int,
        learning_rate: int,
        num_steps: int,
        valid_interval: int,
    ):
        """
        Train the Platt scaler.

        Parameters
        ----------
        train_probabilities: torch.FloatTensor
            Sequence likelihoods from the LLM used as input to the scaler.
        train_targets: torch.FloatTensor
            Desired targets.
        valid_probabilities: torch.FloatTensor
            Sequence likelihoods from the LLM used for calibration.
        valid_targets: torch.FloatTensor
            Desired targets, but for the validation samples.
        batch_size: int
            Batch size for the scaler.
        learning_rate: float
            Learning rate for the scaler.
        num_steps: int
            Number of steps for the scaler.
        valid_interval: int
            Validation interval.
        """
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        train_dataloader = DataLoader(
            TensorDataset(train_probabilities, train_targets), batch_size=batch_size
        )
        valid_dataloader = DataLoader(
            TensorDataset(valid_probabilities, valid_targets), batch_size=batch_size
        )
        loss_func = nn.MSELoss()

        best_scale, best_bias = self.scale.clone(), self.bias.clone()
        best_val_loss = float("inf")

        with tqdm(total=num_steps) as progress_bar:
            for i, (inputs, targets) in tqdm(
                enumerate(loop_dataloader(train_dataloader)), total=num_steps
            ):
                if i > num_steps:
                    break

                outputs = self.forward(inputs)
                loss = loss_func(outputs, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.set_description(
                    f"[Step {i+1}] Loss: {loss.detach().cpu().item():.4f}"
                )
                progress_bar.update(1)

                if i != 0 and (i + 1) % valid_interval == 0:
                    val_loss = 0

                    with torch.no_grad():
                        for j, (inputs, targets) in enumerate(valid_dataloader):
                            outputs = self.forward(inputs)
                            loss = loss_func(outputs, targets)
                            val_loss += loss.item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_scale, best_bias = self.scale.clone(), self.bias.clone()

                    print(f"[Step {i+1}] Validation Loss: {val_loss}")

            self.scale, self.bias = nn.Parameter(best_scale), nn.Parameter(best_bias)


def compute_baselines(
    baselines_methods: List[str],
    model_identifier: str,
    dataset_name: str,
    num_in_context_samples: int,
    data_dir: str,
    # Platt scaling arguments
    platt_scaling_batch_size: Optional[int] = None,
    platt_scaling_learning_rate: Optional[int] = None,
    platt_scaling_num_steps: Optional[int] = None,
    platt_scaling_valid_interval: Optional[int] = None,
    # Miscellaneous arguments
    img_dir: Optional[str] = None,
    result_dir: Optional[str] = None,
    wandb_run: Optional[WandBRun] = None,
):
    """
    Compute the baselines for the QA experiments.

    Parameters
    ----------
    baselines_methods: List[str]
        List of the baselines to compute.
    model_identifier: str
        Identifier of the model whose data we should access.
    dataset_name: str
        Dataset results should be computed for.
    num_in_context_samples: int
        Number of in-context samples that were used with the model.
    data_dir: str
        Path to directory containing the data.
    platt_scaling_batch_size: Optional[int]
        Batch size used for Platt scaling. Defaults to None.
    platt_scaling_learning_rate: Optional[int]
        Learning rate for Platt scaling. Defaults to None.
    platt_scaling_num_steps: Optional[int]
        Number of steps used for Platt scaling. Defaults to None.
    platt_scaling_valid_interval: Optional[int]
        Size of interval used for validation for Platt scaling. Defaults to None.
    img_dir: Optional[str]
        Directory to plot reliability diagrams to. Defaults to None.
    result_dir: Optional[str]
        Directory to save baseline results to. Defaults to None.
    wandb_run: Optional[WandBRun]
        Weights & Biases run to log results. Defaults to None.
    """
    data_dir = os.path.join(
        data_dir,
        dataset_name,
        model_identifier.replace("/", "_"),
        "calibration_data",
        f"in_context_{num_in_context_samples}",
    )
    baseline_results_dir = None
    if result_dir is not None:
        baseline_results_dir = os.path.join(
            result_dir,
            dataset_name,
            model_identifier.replace("/", "_"),
            f"in_context_{num_in_context_samples}",
        )

    # Load data
    split_names = list(DATASET_SPLIT_SIZES[dataset_name].keys())

    if any(
        [
            not os.path.exists(os.path.join(data_dir, f"calibration_data_{split}.dill"))
            for split in split_names
        ]
    ):
        raise FileNotFoundError(
            "Some of the necessary files have not been found. Please execute run_regression_experiment.py first."
        )

    else:
        split_calibration_data = {}

        for split in split_names:
            with open(
                os.path.join(data_dir, f"calibration_data_{split}.dill"), "rb"
            ) as calibration_file:
                split_data = dill.load(calibration_file)

                if "included_questions" in split_data:
                    del split_data["included_questions"]

                split_calibration_data[split] = split_data

    # Do temperature scaling
    temperature_scalers = {}
    for method in ["ts_seq_likelihood", "ts_cot_seq_likelihood"]:
        if method in baselines_methods:
            # Concretely, we obtain temperature scaling parameters by optimizing two scalars:
            # One bias and one shift parameter s.t. the BCE loss on the validation split is minimized.
            train_likelihoods = np.array(
                [
                    question_data[method.replace("ts_", "")]
                    for question_data in split_calibration_data["train"].values()
                ]
            )
            train_likelihoods[np.isnan(train_likelihoods)] = 0
            train_correctness = [
                question_data["accuracy"]
                for question_data in split_calibration_data["train"].values()
            ]
            valid_likelihoods = np.array(
                [
                    question_data[method.replace("ts_", "")]
                    for question_data in split_calibration_data["valid"].values()
                ]
            )
            valid_likelihoods[np.isnan(valid_likelihoods)] = 0
            valid_correctness = [
                question_data["accuracy"]
                for question_data in split_calibration_data["valid"].values()
            ]

            # Compute targets
            train_target_func = get_target_function(
                train_likelihoods, train_correctness
            )
            train_likelihoods = torch.FloatTensor(train_likelihoods)
            train_targets = torch.FloatTensor(train_target_func(train_likelihoods))

            valid_target_func = get_target_function(
                valid_likelihoods, valid_correctness
            )
            valid_likelihoods = torch.FloatTensor(valid_likelihoods)
            valid_targets = torch.FloatTensor(valid_target_func(valid_likelihoods))

            print(f"Train Platt scaler for {method}")
            scaler = PlattScaler()
            scaler.train_scaler(
                train_probabilities=train_likelihoods,
                train_targets=train_targets,
                valid_probabilities=valid_likelihoods,
                valid_targets=valid_targets,
                batch_size=platt_scaling_batch_size,
                learning_rate=platt_scaling_learning_rate,
                num_steps=platt_scaling_num_steps,
                valid_interval=platt_scaling_valid_interval,
            )

            temperature_scalers[method] = scaler

    # ### Compute baseline results ###
    baseline_confidences = defaultdict(dict)
    baselines_results = {}
    masks = defaultdict(dict)

    for method in baselines_methods:
        for split_name in split_names:
            if "test" not in split_name:  # Only evaluate test splits
                continue

            split_data = split_calibration_data[split_name]

            # Extract confidences
            likelihoods = np.array(
                [
                    question_data["seq_likelihood"]
                    for question_data in split_data.values()
                ]
            )
            likelihoods[np.isnan(likelihoods)] = 0
            baseline_confidences[split_name]["seq_likelihood"] = likelihoods

            cot_likelihoods = np.array(
                [
                    question_data["cot_seq_likelihood"]
                    for question_data in split_data.values()
                ]
            )
            cot_likelihoods[np.isnan(cot_likelihoods)] = 0
            baseline_confidences[split_name]["cot_seq_likelihood"] = cot_likelihoods

            # Do temperature scaling
            if method in ["ts_seq_likelihood", "ts_cot_seq_likelihood"]:
                scaler = temperature_scalers[method]

                if method == "ts_seq_likelihood":
                    inputs = torch.FloatTensor(likelihoods)

                else:
                    inputs = torch.FloatTensor(cot_likelihoods)

                with torch.no_grad():
                    outputs = scaler.forward(inputs).numpy()

                baseline_confidences[split_name][method] = outputs

            # Convert verbalized uncertainties into confidence scores
            infix = "_cot" if "cot" in method else ""
            if "qual" in method:
                qual_uncertainties = [
                    question_data[f"verbalized{infix}_qual"]
                    for question_data in split_data.values()
                ]
                confidences, successes = extract_verbalized_confidence(
                    qual_uncertainties,
                    mode="qualitative",
                    expression_mapping=QUALITATIVE_SCALE,
                )
                baseline_confidences[split_name][
                    f"verbalized{infix}_qual"
                ] = confidences
                masks[split_name][f"verbalized{infix}_qual"] = np.array(successes)

            elif "quant" in method:
                quant_uncertainties = [
                    question_data[f"verbalized{infix}_quant"]
                    for question_data in split_data.values()
                ]
                confidences, successes = extract_verbalized_confidence(
                    quant_uncertainties, mode="quantitative"
                )
                baseline_confidences[split_name][
                    f"verbalized{infix}_quant"
                ] = confidences
                masks[split_name][f"verbalized{infix}_quant"] = np.array(successes)

    eval_data = defaultdict(lambda: dict())
    for split_name in split_names:
        for baseline_name, confidences in baseline_confidences[split_name].items():
            correctness = np.array(
                [
                    question_data["accuracy"]
                    for question_data in split_calibration_data[split_name].values()
                ]
            )

            if baseline_name in masks[split_name]:
                baseline_mask = masks[split_name][baseline_name]
                baselines_results[f"{split_name}_{baseline_name}_success"] = np.mean(
                    baseline_mask.astype(int)
                )

                correctness = correctness[baseline_mask]

            baseline_res = evaluate_confidences(
                split_name=split_name,
                add_name=baseline_name,
                all_confidences=confidences,
                all_correctness=correctness,
            )
            baselines_results.update(baseline_res)
            eval_data[baseline_name][split_name] = {
                "all_confidences": confidences,
                "all_correctness": correctness,
            }

            # Plot reliability diagram
            if img_dir is not None:
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)

                plot_reliability_diagram(
                    confidences,
                    correctness,
                    save_path=os.path.join(
                        img_dir, f"{split_name}_{baseline_name}.png"
                    ),
                    success_percentage=baselines_results.get(
                        f"{split_name}_{baseline_name}_success", 1
                    ),
                )

    # Save results
    if baseline_results_dir is not None:
        for baseline_name, baseline_data in eval_data.items():
            with open(
                os.path.join(
                    baseline_results_dir,
                    f"{timestamp}_{baseline_name}_results.dill",
                ),
                "wb",
            ) as results_file:
                dill.dump(
                    {
                        "info": {
                            "timestamp": timestamp,
                            "dataset_name": dataset_name,
                            "model_identifier": model_identifier,
                            "baseline_name": baseline_name,
                        },
                        "eval_data": baseline_data,
                    },
                    results_file,
                )

        results_df = pd.DataFrame(columns=EVAL_METRIC_ORDER)

        # This is an inefficient way to create the results dataframe, but we do not have so many entries so whatever
        test_splits = [
            split for split in split_calibration_data.keys() if "test" in split
        ]
        for baseline_name in baselines_methods:
            for name, result in baselines_results.items():
                for eval_metric in EVAL_METRIC_ORDER:
                    if f"_{eval_metric}" in name and any(
                        [f"{split}_{baseline_name}" in name for split in test_splits]
                    ):
                        results_df.at[baseline_name, eval_metric] = round(result, 2)
                        break

        print(results_df)
        print(results_df.to_latex(float_format="%.2f"))
        for name, result in baselines_results.items():
            print(f"{name}: {result:.2f}")

        if wandb_run is not None:
            wandb.log(baselines_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baselines-methods",
        nargs="+",
        choices=BASELINES_METHODS,
        default=BASELINES_METHODS,
    )
    parser.add_argument(
        "--model-identifier",
        type=str,
        default=MODEL_IDENTIFIER,
        help="OpenAI identifier for model.",
    )
    parser.add_argument(
        "--dataset-name", type=str, help="Name of the dataset.", choices=DATASETS
    )
    parser.add_argument(
        "--num-in-context-samples", type=int, default=NUM_IN_CONTEXT_SAMPLES
    )
    parser.add_argument(
        "--temp-scaling-batch-size",
        type=int,
        default=PLATT_SCALING_BATCH_SIZE,
    )
    parser.add_argument(
        "--temp-scaling-learning-rate",
        type=float,
        default=PLATT_SCALING_LEARNING_RATE,
    )
    parser.add_argument(
        "--temp-scaling-num-steps", type=int, default=PLATT_SCALING_NUM_STEPS
    )
    parser.add_argument(
        "--temp-scaling-valid-interval", type=int, default=PLATT_SCALING_VALID_INTERVAL
    )
    parser.add_argument(
        "--data-dir", type=str, default=DATA_DIR, help="Directory containing data."
    )
    parser.add_argument(
        "--img-dir", type=str, help="Directory to plot images into.", default=None
    )
    parser.add_argument(
        "--result-dir", type=str, help="Directory to save results to.", default=None
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Whether to track experiments via Weights & Biases.",
    )

    args = parser.parse_args()
    timestamp = str(datetime.now().strftime("%d-%m-%Y_(%H:%M:%S)"))

    wandb_run = None
    if args.wandb:
        wandb_run = wandb.init(
            project=PROJECT_NAME,
            tags=[args.dataset_name, args.model_identifier],
            settings=wandb.Settings(start_method="fork"),
            config={
                "model_identifier": args.model_identifier,
                "dataset_name": args.dataset_name,
                "num_in_context_samples": args.num_in_context_samples,
                "timestamp": timestamp,
            },
            name=args.wandb_name,
        )

    compute_baselines(
        baselines_methods=args.baselines_methods,
        model_identifier=args.model_identifier,
        dataset_name=args.dataset_name,
        num_in_context_samples=args.num_in_context_samples,
        data_dir=args.data_dir,
        platt_scaling_batch_size=args.temp_scaling_batch_size,
        platt_scaling_learning_rate=args.temp_scaling_learning_rate,
        platt_scaling_num_steps=args.temp_scaling_num_steps,
        platt_scaling_valid_interval=args.temp_scaling_valid_interval,
        result_dir=args.result_dir,
        wandb_run=wandb_run,
        img_dir=args.img_dir,
    )

    if wandb_run is not None:
        wandb_run.finish()
