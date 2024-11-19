"""
Perform hyperparameter search.
"""

# STD
import argparse
from datetime import datetime
import dill
import json
import os
from typing import Optional, List, Dict, Any
import warnings

# EXT
from codecarbon import OfflineEmissionsTracker
from knockknock import telegram_sender
import numpy as np
from relplot.metrics import smECE_slow as smece
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import transformers
from transformers import AutoModelForSequenceClassification
import wandb
from wandb.sdk.wandb_run import Run as WandBRun

# PROJECT
from src.constants import (
    CALIBRATION_MODEL_IDENTIFIER,
    CALIBRATION_MODEL_PARAMS,
    NUM_IN_CONTEXT_SAMPLES,
    DATASET_SPLIT_SIZES,
    DATA_DIR,
    PROJECT_NAME,
    SEED,
    EMISSION_DIR,
)
from src.eval import ece


try:
    from secret import TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID, COUNTRY_CODE

    SECRET_IMPORTED = True

except (ImportError, ModuleNotFoundError) as _:
    try:
        TELEGRAM_API_TOKEN = os.environ["TELEGRAM_API_TOKEN"]
        TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
        COUNTRY_CODE = os.environ["COUNTRY_CODE"]
        SECRET_IMPORTED = True

    except AttributeError:
        warnings.warn(
            "secret.py wasn't found, please rename secret_template.py and fill in the information or make variables "
            "available through os.environ."
        )


def perform_hyperparameter_search(
    model_identifier: str,
    calibration_model_identifier: str,
    model_params: Dict[str, Any],
    num_in_context_samples: int,
    dataset_name: str,
    use_binary_targets: bool,
    data_dir: str,
    device: torch.device | str = "cpu",
    seed: Optional[int] = None,
    wandb_run: Optional[WandBRun] = None,
) -> str:
    """
    Perform hyperparameter search for a list of models and save the results into a directory.

    Parameters
    ----------
    model_identifier: str
        Identifier of the target LLM.
    calibration_model_identifier: str
        Identifier of the calibration model.
    model_params: Dict[str, Any]
        Training hyperparameters for the current attempt.
    num_in_context_samples: int
        Number of in context samples.
    dataset_name: str
        Name of data set models should be evaluated on.
    use_binary_targets: bool
        Whether the calibrator is supposed to use binary targets instead of clustering based ones.
    data_dir: str
        Directory the data is stored in.
    device: torch.device | str
        Device hyperparameter search happens on.
    seed: Optional[int]
        Seed for the hyperparameter run.
    wandb_run: Optional[WandBRun]
        Weights and Biases Run to track training statistics. Training and validation loss (if applicable) are tracked by
        default, everything else is defined in _epoch_iter() and _finetune() depending on the model.

    Returns
    -------
    str
        Information being passed on to knockknock.
    """
    if seed is not None:
        torch.random.manual_seed(seed)

    info_dict = {}

    if wandb_run is not None:
        info_dict["config"] = wandb_run.config.as_dict()

    # Define calibration dataloader path
    calibration_data_dir = os.path.join(
        data_dir,
        dataset_name,
        model_identifier.replace("/", "_"),
        "calibration_data",
        f"in_context_{num_in_context_samples}",
    )
    calibration_target_file = os.path.join(
        calibration_data_dir, "calibration_targets.dill"
    )

    dataset_split_names = list(DATASET_SPLIT_SIZES[dataset_name].keys())

    # Read data and build data splits
    if any(
        [
            *[
                not os.path.exists(
                    os.path.join(
                        calibration_data_dir,
                        f"calibration_data_answer_question_{split}.dl",
                    )
                )
                for split in dataset_split_names
            ],
            not os.path.exists(calibration_target_file),
        ]
    ):
        raise FileNotFoundError(
            "Some of the necessary files have not been found. Please execute run_regression_experiment.py first."
        )

    # Load dataloaders and calibration targets
    with open(calibration_target_file, "rb") as calibration_file:
        calibration_targets = dill.load(calibration_file)

    calibration_dataloaders = {
        split: torch.load(
            os.path.join(
                os.path.join(
                    calibration_data_dir, f"calibration_data_answer_question_{split}.dl"
                )
            )
        )
        for split in dataset_split_names
    }

    loss_weights = None
    if use_binary_targets:
        all_labels = []

        for batch in calibration_dataloaders["train"]:
            all_labels += list(batch["correctness"].numpy())

        loss_weights = torch.FloatTensor(
            compute_class_weight(class_weight="balanced", classes=[0, 1], y=all_labels)
        ).to(device)

    # ### MODEL TRAINING ###
    try:
        warmup_fraction = model_params["warmup_fraction"]
        num_training_steps = model_params["num_training_steps"]

        # Load calibration model
        calibration_model = AutoModelForSequenceClassification.from_pretrained(
            calibration_model_identifier
        )
        calibration_model = calibration_model.to(device)

        def loop_dataloader(dataloader):
            while True:
                for batch in dataloader:
                    yield batch

        # ### TRAINING LOOP ###
        optimizer = optim.AdamW(
            calibration_model.parameters(),
            lr=model_params["learning_rate"],
            weight_decay=model_params["weight_decay"],
        )
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup_fraction * num_training_steps),
            num_training_steps=num_training_steps,
        )

        for i, batch in enumerate(loop_dataloader(calibration_dataloaders["train"])):
            if i >= num_training_steps:
                break

            input_ids = batch["input_ids"].squeeze(1).to(device)
            attention_mask = batch["attention_mask"].squeeze(1).to(device)
            if use_binary_targets:
                targets = batch["correctness"].to(device)
                outputs = calibration_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                preds = outputs.logits
                weights = loss_weights[targets].unsqueeze(-1)
                targets = F.one_hot(targets, 2).float()
                loss_func = nn.BCEWithLogitsLoss(weight=weights)

            else:
                targets = torch.FloatTensor(
                    [
                        calibration_targets[question_id]
                        for question_id in batch["question_id"]
                    ]
                ).to(device)
                outputs = calibration_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                preds = F.sigmoid(outputs.logits[:, 1])
                loss_func = nn.MSELoss()

            loss = loss_func(preds, targets)
            loss.backward()
            clip_grad_norm_(
                calibration_model.parameters(),
                max_norm=model_params["num_training_steps"],
            )

            print(
                f"[Step {i + 1}/{num_training_steps}] Loss: {loss.detach().cpu().item():.4f}"
            )

            if wandb_run is not None:
                wandb_run.log({"loss": loss.detach().cpu().item()})

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Evaluate on calibration set
            if (
                i > 0 and i % model_params["eval_interval"] == 0
            ) or i == num_training_steps - 1:
                val_correctness = []
                val_confidences = []
                val_targets = []
                val_losses = []

                with torch.no_grad():
                    for batch in calibration_dataloaders["valid"]:
                        input_ids = batch["input_ids"].squeeze(1).to(device)
                        attention_mask = batch["attention_mask"].squeeze(1).to(device)
                        if use_binary_targets:
                            targets = batch["correctness"].to(device)
                            val_targets += targets.cpu().tolist()
                            outputs = calibration_model(
                                input_ids=input_ids, attention_mask=attention_mask
                            )
                            preds = F.softmax(outputs.logits, dim=-1)
                            weights = loss_weights[targets].unsqueeze(-1)
                            targets = (
                                F.one_hot(batch["correctness"], 2).float().to(device)
                            )
                            loss_func = nn.BCEWithLogitsLoss(weight=weights)
                            val_loss = loss_func(preds, targets).cpu().item()
                            preds = preds[:, 1]

                        else:
                            val_targets += [
                                calibration_targets[question_id]
                                for question_id in batch["question_id"]
                            ]
                            targets = torch.FloatTensor(
                                [
                                    calibration_targets[question_id]
                                    for question_id in batch["question_id"]
                                ]
                            ).to(device)
                            outputs = calibration_model(
                                input_ids=input_ids, attention_mask=attention_mask
                            )
                            preds = F.sigmoid(outputs.logits[:, 1])
                            loss_func = nn.MSELoss()
                            val_loss = loss_func(preds, targets).cpu().item()

                        val_losses.append(val_loss)
                        val_confidences += preds.cpu().tolist()
                        val_correctness += batch["correctness"].tolist()

                val_metrics = {
                    "validation_loss": np.mean(val_losses),
                    "validation_ece": ece(y_true=val_targets, y_pred=val_confidences),
                    "validation_smece": smece(
                        f=np.array(val_confidences), y=np.array(val_targets)
                    ),
                    "validation_bier_score": brier_score_loss(
                        y_true=val_correctness, y_prob=val_confidences
                    ),
                    "validation_auroc": roc_auc_score(
                        y_true=val_correctness, y_score=val_confidences
                    ),
                }
                print(f"[Step: {i + 1}] Validation results:")
                print("\n".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))

                if wandb_run is not None:
                    wandb_run.log(val_metrics)

    # In case of nans due bad training parameters
    except (ValueError, RuntimeError) as e:
        print(f"There was an error: '{str(e)}', run aborted.")

    if wandb_run is not None:
        info_dict["url"] = wandb.run.get_url()

    if tracker is not None:
        tracker.stop()
        emissions = tracker._prepare_emissions_data().emissions
        info_dict["emissions"] = emissions

        if wandb_run is not None:
            wandb_run.log({"emissions": emissions})

    return "\n" + json.dumps(info_dict, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Dataset to run experiments on.",
    )
    parser.add_argument(
        "--model-identifier",
        type=str,
        help="Huggingface Hub identifier for model.",
    )
    parser.add_argument(
        "--use-binary-targets",
        action="store_true",
        default=False,
        help="Whether to use binary targets instead of clustering targets.",
    )
    parser.add_argument(
        "--calibration-model-identifier",
        type=str,
        help="Huggingface Hub identifier for model.",
        default=CALIBRATION_MODEL_IDENTIFIER,
    )
    parser.add_argument(
        "--num-in-context-samples", type=int, default=NUM_IN_CONTEXT_SAMPLES
    )
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--track-emissions", action="store_true", default=False)
    parser.add_argument("--emission-dir", type=str, default=EMISSION_DIR)
    parser.add_argument("--knock", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--notes", type=str, default=False, help="Additional notes for the experiment."
    )

    # Parse into the arguments specified above, everything else are ran parameters
    args, config = parser.parse_known_args()
    print(args)

    def _stupid_parse(raw_config: List[str]):
        config = {}

        for raw_arg in raw_config:
            try:
                arg, value = raw_arg.strip().replace("--", "").split("=")
                arg = arg.replace("-", "_")
            except ValueError:
                continue

            try:
                config[arg] = int(value)

            except ValueError:
                try:
                    config[arg] = float(value)

                except ValueError:
                    # Argument is probably a string
                    config[arg] = value

        return config

    config = _stupid_parse(config)
    model_params = dict(CALIBRATION_MODEL_PARAMS)
    model_params.update(config)

    print(model_params)

    tracker = None
    wandb_run = None
    timestamp = str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
    wandb_run = wandb.init(
        project=PROJECT_NAME,
        tags=[args.dataset_name, args.calibration_model_identifier],
        settings=wandb.Settings(start_method="fork"),
        config={
            "calibration_model_identifier": args.calibration_model_identifier,
            "dataset_name": args.dataset_name,
            "num_in_context_samples": args.num_in_context_samples,
            "timestamp": timestamp,
        },
    )

    if args.track_emissions:
        timestamp = str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
        emissions_path = os.path.join(args.emission_dir, timestamp)
        os.makedirs(emissions_path)
        tracker = OfflineEmissionsTracker(
            project_name="nlp_uncertainty_zoo-hyperparameters",
            country_iso_code=COUNTRY_CODE,
            output_dir=emissions_path,
        )
        tracker.start()

    # Apply decorator
    if args.knock:
        if not SECRET_IMPORTED:
            raise ImportError(
                "secret.py wasn't found, please rename secret_template.py and fill in the information."
            )

        perform_hyperparameter_search = telegram_sender(
            token=TELEGRAM_API_TOKEN, chat_id=TELEGRAM_CHAT_ID
        )(perform_hyperparameter_search)

    perform_hyperparameter_search(
        model_identifier=args.model_identifier,
        calibration_model_identifier=args.calibration_model_identifier,
        model_params=model_params,
        num_in_context_samples=args.num_in_context_samples,
        dataset_name=args.dataset_name,
        use_binary_targets=args.use_binary_targets,
        data_dir=args.data_dir,
        device=args.device,
        seed=args.seed,
        wandb_run=wandb_run,
    )

    if tracker is not None:
        tracker.stop()

    if wandb_run is not None:
        wandb_run.finish()
