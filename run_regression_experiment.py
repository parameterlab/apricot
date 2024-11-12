"""
Main experimental script for single prompt suffix experiments.
"""

# STD
import argparse
from copy import deepcopy
import dill
import gc
import os
from typing import Optional, List, Dict, Tuple, Any
import warnings

# EXT
from codecarbon import OfflineEmissionsTracker
from datetime import datetime
from knockknock import telegram_sender
import numpy as np
from relplot.metrics import smECE_slow as smece
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
import transformers
import wandb
from wandb.sdk.wandb_run import Run as WandBRun

# PROJECT
from src.calibration import (
    extract_model_calibration_data,
    compute_question_calibration_targets,
)
from src.data import preprocess_dataset
from src.constants import (
    ALLOWED_INPUTS,
    BLACK_BOX_MODELS,
    CALIBRATION_MODEL_IDENTIFIER,
    DATASETS,
    NUM_IN_CONTEXT_SAMPLES,
    BATCH_SIZE,
    CACHE_DIR,
    CALIBRATOR_BATCH_SIZE,
    DATA_DIR,
    DATASET_SPLIT_SIZES,
    EMISSION_DIR,
    EVAL_INTERVAL,
    INPUT_PARTS,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MODEL_IDENTIFIER,
    NUM_TRAINING_STEPS,
    PROJECT_NAME,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    SEED,
    RESULT_DIR,
)
from src.eval import ece, evaluate_model
from src.plotting import plot_reliability_diagram
from src.prompts import (
    QUAL_VERBALIZED_CONFIDENCE_PROMPT,
    QUANT_VERBALIZED_CONFIDENCE_PROMPT,
)
from src.utils import unpack_dataloader, create_calibration_dataloader, loop_dataloader

# Knockknock support
SECRET_IMPORTED = False
try:
    from secret import TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID, COUNTRY_CODE, WANDB_API_KEY

    SECRET_IMPORTED = True
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

except (ImportError, ModuleNotFoundError) as e:
    warnings.warn("secret.py could not be imported.")

    try:
        TELEGRAM_API_TOKEN = os.environ["TELEGRAM_API_TOKEN"]
        TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
        COUNTRY_CODE = os.environ["COUNTRY_CODE"]
        WANDB_API_KEY = os.environ["COUNTRY_CODE"]
        SECRET_IMPORTED = True

    except AttributeError:
        raise ImportError(
            "secret.py wasn't found, please rename secret_template.py and fill in the information or make variables "
            "available through os.environ."
        )


# CUDA
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# HF
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR


def create_or_load_calibration_data(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data_loader: DataLoader,
    device: torch.device | str,
    max_samples: int,
    data_dir: str,
    data_path: str,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Create calibration data or, if it already exists, load it from disk.

    Parameters
    ----------
    model: AutoModelForCausalLM
        Target LLM.
    tokenizer: AutoTokenizer
        Tokenizer for target model.
    data_loader: DataLoader
        Dataloader of split the data should be extracted from.
    device: torch.device | str
        Device the target model lives on.
    max_samples: int
        Maximum number of samples the calibration data should be extracted for.
    data_dir: str
        Parent directory for data.
    data_path: str
        Full path to calibration data file.

    Returns
    -------
    Tuple[Dict[str, Dict[str, Any]], List[str]]
        All the relevant data for calibration extracted from the model, including its answer, correctness and
        confidence. Additionally, return a list of all question ids that have been processed.
    """
    if not os.path.exists(data_path):
        calibration_data, included_questions = extract_model_calibration_data(
            model=model,
            tokenizer=tokenizer,
            calibration_split=data_loader,
            device=device,
            max_samples=max_samples,
        )
        # Add it to the dict so we only have to pickle a single object
        calibration_data["included_questions"] = included_questions

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        with open(data_path, "wb") as calibration_file:
            dill.dump(calibration_data, calibration_file)

        del calibration_data["included_questions"]  # Delete it after saving

    else:
        with open(data_path, "rb") as calibration_file:
            calibration_data = dill.load(calibration_file)

        # Extract included questions from the pickled object
        included_questions = calibration_data["included_questions"]
        del calibration_data["included_questions"]

    return calibration_data, included_questions


def run_single_calibration_experiment(
    model_identifier: str,
    calibration_model_identifier: str,
    dataset_name: str,
    num_training_steps: int,
    num_in_context_samples: int,
    batch_size: int,
    calibrator_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_fraction: float,
    max_grad_norm: float,
    eval_interval: int,
    use_binary_targets: bool,
    input_parts: List[str],
    device: str,
    data_dir: str,
    result_dir: str,
    seed: int,
    model_save_dir: Optional[str] = None,
    wandb_run: Optional[WandBRun] = None,
):
    """
    Run experiments which train a single prompt suffix to improve model calibration.

    Parameters
    ----------
    model_identifier: str
        Identifier of the target model from the Huggingface hub to use.
    calibration_model_identifier: str
        Identifier for the calibration model to use.
    dataset_name: str
        Name of the dataset to use.
    num_training_steps: int
        Number of finetuning steps for the calibration model.
    num_in_context_samples: int
        Number of in-context samples to add to the prompt.
    batch_size: int
        Batch size used for the target model.
    calibrator_batch_size: int
        Batch size used for the calibration model.
    learning_rate: float
        Learning rate used for finetuning the calibration model.
    weight_decay: float
        Weight decay used for finetuning the calibration model.
    warmup_fraction: float
        Percentage of training steps used as warmup for the learning rate scheduler.
    max_grad_norm: float
        Maximum gradient norm for finetuning the calibration model.
    eval_interval: int
        Interval at which to evaluate the calibrator on the validation set.
    use_binary_targets: bool
        Whether to train the calibrator with binary targets.
    input_parts: List[str]
        Which parts of the input to use. This includes 'question', 'answer', 'cot_answer', 'quantitative',
        'cot_quantitative', 'qualitative' and 'cot_qualitative'. This is used for ablation purposes and to see whether
        exposing the calibrator to more features from the target LLM improves performance.
        These parts cannot be combined arbitrarily: 'question' is always needed, followed by the answer (CoT or normal)
        and the verbalized uncertainty, either qualitative or quantitative in the normal or CoT version.
    device: str
        Devices the models live on.
    data_dir: str
        Directory containing datasets.
    result_dir: str
        Directory to save results into.
    seed: int
        Random seed used for replicability.
    model_save_dir: Optional[str]
        Directory to save model weights to. Used to e.g. push models to the HF hub.
    wandb_run: WandBRun
        Weights & Biases run for logging.
    """
    # Validate input parts - this defined what is given to the auxiliary model as input.
    assert (
        len(set(input_parts) - ALLOWED_INPUTS) == 0
    ), "Unrecognized arguments found in 'input_parts'."
    assert (
        "question" in input_parts
    ), "'input_parts' always has to contain at least 'question'."
    if any([part in input_parts for part in ["qualitative", "quantitative"]]):
        assert "question" in input_parts and (
            "answer" in input_parts or "cot_answer" in input_parts
        ), "Given input parts require 'question' and 'answer' to be in 'input_parts'."
    assert (
        "answer" not in input_parts or "cot_answer" not in input_parts
    ), "Choose either 'answer' or 'cot_answer' for 'input_parts', not both."
    assert (
        "qualitative" not in input_parts or "quantitative" not in input_parts
    ), "Choose either 'qualitative' or 'cot_answer' for 'quantitative', not both."

    input_parts = list(sorted(input_parts))
    suffix = "_".join(input_parts)

    torch.manual_seed(seed)
    np.random.seed(seed)

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

    # #### Step 1: Create calibration data ####
    # This involves preprocessing the given datasets, feeding them through the (local) target LLM, and saving the
    # resulting data used to train the auxiliary model as .dill files. If we use a black-box model, this step is
    # performed seperately in get_openai_data.py, and requires this script to have been run first.

    # Load calibration model
    calibration_tokenizer = AutoTokenizer.from_pretrained(calibration_model_identifier)
    calibration_config = AutoConfig.from_pretrained(calibration_model_identifier)

    # Check whether calibration data is available, otherwise generate
    dataset_split_names = list(DATASET_SPLIT_SIZES[dataset_name].keys())

    if (
        any(
            [
                not os.path.exists(
                    os.path.join(
                        calibration_data_dir,
                        f"calibration_data_{split}.dill",
                    )
                )
                for split in dataset_split_names
            ]
        )
        and model_identifier not in BLACK_BOX_MODELS
    ):
        # Pre-process data from scratch or load from disk, collect target models responses
        data_loaders = preprocess_dataset(
            model_identifier=model_identifier,
            dataset_name=dataset_name,
            num_in_context_samples=num_in_context_samples,
            batch_size=batch_size,
            data_dir=data_dir,
        )

        # Load model
        config = AutoConfig.from_pretrained(model_identifier)
        config.max_length = 550
        model = AutoModelForCausalLM.from_pretrained(
            model_identifier,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            config=config,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_identifier, padding_side="left")

        # Unpack dataloaders
        inputs_, question_ids = {}, {}
        calibration_data, included_questions = {}, {}

        for split in dataset_split_names:
            inputs_[split], question_ids[split] = unpack_dataloader(
                data_loaders[split], tokenizer=tokenizer
            )

            calibration_data_path = os.path.join(
                calibration_data_dir, f"calibration_data_{split}.dill"
            )

            (
                calibration_data[split],
                included_questions[split],
            ) = create_or_load_calibration_data(
                model=model,
                tokenizer=tokenizer,
                data_loader=data_loaders[split],
                device=device,
                max_samples=DATASET_SPLIT_SIZES[dataset_name][split],
                data_path=calibration_data_path,
                data_dir=calibration_data_dir,
            )

        # Free up memory from preprocessing - model will be reloaded within workers
        del model, tokenizer, data_loaders
        torch.cuda.empty_cache()
        gc.collect()

    # Load calibration data from disk
    else:
        calibration_data = {}

        for split_name in dataset_split_names:
            calibration_data_path = os.path.join(
                calibration_data_dir, f"calibration_data_{split_name}.dill"
            )
            with open(calibration_data_path, "rb") as calibration_file:
                calibration_data[split_name] = dill.load(calibration_file)

    # Generate calibration targets or load them from disk
    if not os.path.exists(calibration_target_file):
        # Compute calibration targets
        # Merge also test data in here
        all_calibration_data = {}
        for split in dataset_split_names:
            all_calibration_data.update(calibration_data[split])

        calibration_targets = compute_question_calibration_targets(
            all_calibration_data,
            data_dir=calibration_data_dir,
        )

        with open(calibration_target_file, "wb") as calibration_file:
            dill.dump(calibration_targets, calibration_file)

    else:
        with open(calibration_target_file, "rb") as calibration_file:
            calibration_targets = dill.load(calibration_file)

    # #### Step 2: Create finetuning data for auxiliary model ####
    # We now load the calibration data from the .dill files and preprocess them to be used for training for the
    # auxiliary model.

    # Create calibration dataloaders, otherwise load them from disk
    calibration_dataloaders = {}
    if any(
        [
            not os.path.exists(
                os.path.join(
                    calibration_data_dir, f"calibration_data_{suffix}_{split}.dl"
                )
            )
            for split in dataset_split_names
        ]
    ):
        for split in dataset_split_names:
            # Filter by question that are contained in the current fraction
            calibration_split_data = calibration_data[split]

            if "included_questions" in calibration_split_data:
                del calibration_split_data["included_questions"]

            filtered_question_ids, filtered_inputs = [], []

            for question_id, question_data in calibration_split_data.items():
                filtered_question_ids.append(question_id)

                # Could be one of: question_only -> question (for ablations) / answer / cot_answer / verbalized_quant /
                # verbalized_qual / verbalized_cot_quant / verbalized_cot_qual

                input_ = question_data["question"]

                # Choose type of answer to include in input
                if "answer" in input_parts:
                    input_ += f" [SEP] {question_data['answer']}"

                elif "cot_answer" in input_parts:
                    input_ += f" [SEP] {question_data['cot_answer']}"

                # Choose type of verbalized uncertainty to include in input
                infix = "_cot" if "cot_answer" in input_parts else ""
                if "qualitative" in input_parts:
                    input_ += (
                        f" [SEP] {QUAL_VERBALIZED_CONFIDENCE_PROMPT} [SEP] "
                        f"{question_data[f'verbalized{infix}_qual']}"
                    )

                elif "quantitative" in input_parts:
                    input_ += (
                        f" [SEP] {QUANT_VERBALIZED_CONFIDENCE_PROMPT} [SEP] "
                        f"{question_data[f'verbalized{infix}_quant']}"
                    )

                filtered_inputs.append(input_)

            # Create separate dataloaders for the calibration model
            calibration_dataloaders[split] = create_calibration_dataloader(
                batch_size=calibrator_batch_size,
                inputs_=filtered_inputs,
                question_ids=filtered_question_ids,
                calibration_data=calibration_data[split],
                calibration_targets=calibration_targets,
                tokenizer=calibration_tokenizer,
                # Tokenizer kwargs
                padding="max_length",
                truncation=True,
                max_length=calibration_config.max_position_embeddings,
                return_tensors="pt",
            )

        # Save dataloader for future runs
        for split, data_loader in calibration_dataloaders.items():
            torch.save(
                data_loader,
                os.path.join(
                    calibration_data_dir, f"calibration_data_{suffix}_{split}.dl"
                ),
            )

    # If everything is pre-computed, just load from disk
    else:
        with open(calibration_target_file, "rb") as calibration_file:
            calibration_targets = dill.load(calibration_file)

        calibration_dataloaders = {
            split: torch.load(
                os.path.join(
                    os.path.join(
                        calibration_data_dir, f"calibration_data_{suffix}_{split}.dl"
                    )
                )
            )
            for split in dataset_split_names
        }

    # Load calibration model
    calibration_model = AutoModelForSequenceClassification.from_pretrained(
        calibration_model_identifier
    )
    calibration_model = calibration_model.to(device)

    # Compute loss weights
    loss_weights = None
    if use_binary_targets:
        all_labels = []

        for batch in calibration_dataloaders["train"]:
            all_labels += list(batch["correctness"].numpy())

        loss_weights = torch.FloatTensor(
            compute_class_weight(class_weight="balanced", classes=[0, 1], y=all_labels)
        ).to(device)

    # ### TRAINING LOOP ###
    optimizer = optim.AdamW(
        list(calibration_model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(int(warmup_fraction * num_training_steps), 100),
        num_training_steps=int(
            num_training_steps * 1.1
        ),  # This makes sure that the final LR isn't just 0
    )
    best_model = None
    best_val_loss = float("inf")

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
            preds = F.softmax(outputs.logits, dim=-1)[:, 1]
            loss_func = nn.MSELoss()

        loss = loss_func(preds, targets)
        loss.backward()
        clip_grad_norm_(calibration_model.parameters(), max_norm=max_grad_norm)

        print(
            f"[Step {i+1}/{num_training_steps}] Loss: {loss.detach().cpu().item():.4f}"
        )

        if wandb_run is not None:
            wandb_run.log({"loss": loss.detach().cpu().item()})

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # Evaluate on calibration set
        if i > 0 and i % eval_interval == 0:
            val_correctness = []
            val_confidences = []
            val_targets = []
            val_loss = 0

            with torch.no_grad():
                for batch in calibration_dataloaders["valid"]:
                    input_ids = batch["input_ids"].squeeze(1).to(device)
                    attention_mask = batch["attention_mask"].squeeze(1).to(device)

                    if use_binary_targets:
                        targets = batch["correctness"].to(device)
                        weights = loss_weights[targets].unsqueeze(-1)
                        val_targets += targets.cpu().tolist()
                        outputs = calibration_model(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        preds = F.softmax(outputs.logits, dim=-1)
                        targets = F.one_hot(batch["correctness"], 2).float().to(device)
                        loss_func = nn.BCEWithLogitsLoss(weight=weights)
                        val_loss += loss_func(preds, targets).cpu().item()
                        preds = preds[:, 1]

                    else:
                        targets = [
                            calibration_targets[question_id]
                            for question_id in batch["question_id"]
                        ]
                        val_targets += targets
                        targets = torch.FloatTensor(targets).to(device)
                        outputs = calibration_model(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        preds = F.softmax(outputs.logits, dim=-1)[:, 1]
                        val_loss += loss_func(preds, targets).cpu().item()

                    val_confidences += preds.cpu().tolist()
                    val_correctness += batch["correctness"].tolist()

            val_metrics = {
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
                "validation_loss": val_loss,
            }
            print(f"[Step: {i+1}] Validation results:")
            print("\n".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                calibration_model = calibration_model.to("cpu")
                best_model = deepcopy(calibration_model)
                calibration_model = calibration_model.to(device)

            if wandb_run is not None:
                wandb_run.log(val_metrics)

    # Model saving
    if model_save_dir is not None:
        model_name = f"{calibration_model_identifier}_for_{model_identifier}_{dataset_name}"
        model_save_path = os.path.join(model_save_dir, model_name)

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        trainer = transformers.Trainer(model=calibration_model)
        trainer.save_model(model_save_path)

    # ### EVALUATION ###
    # Compute ECE, Brier score, accuracy, AUROC
    metrics, eval_data = evaluate_model(
        calibration_model=calibration_model,
        dataloaders=calibration_dataloaders,
        calibration_targets=calibration_targets,
        device=device,
        use_binary_targets=use_binary_targets,
    )
    print(metrics)

    # Save results
    timestamp = str(datetime.now().strftime("%d-%m-%Y_(%H:%M:%S)"))
    model_results_dir = os.path.join(
        result_dir,
        dataset_name,
        model_identifier.replace("/", "_"),
        f"in_context_{num_in_context_samples}",
    )

    if not os.path.exists(model_results_dir):
        os.makedirs(model_results_dir)

    # Add suffix to distinguish different variants
    suffix = ""

    if use_binary_targets:
        suffix += "binary_"

    suffix += "_".join(input_parts)

    with open(
        os.path.join(model_results_dir, f"{timestamp}_{suffix}_results.dill"),
        "wb",
    ) as results_file:
        dill.dump(
            {
                "info": {
                    "timestamp": timestamp,
                    "dataset_name": dataset_name,
                    "model_identifier": model_identifier,
                    "num_training_steps": num_training_steps,
                    "use_binary_targets": use_binary_targets,
                    "input_parts": input_parts,
                    "suffix": suffix,
                },
                "eval_data": eval_data,
            },
            results_file,
        )

    # Create plot
    for split_name, split_eval_data in eval_data.items():
        all_confidences = split_eval_data["all_confidences"]
        all_correctness = split_eval_data["all_correctness"]
        plot_reliability_diagram(
            all_confidences=all_confidences,
            all_correctness=all_correctness,
            save_path=os.path.join(
                model_results_dir,
                f"{timestamp}_{split_name}_{suffix}.png",
            ),
        )

    metrics = {f"{key}": value for key, value in metrics.items()}
    print(metrics)

    if wandb_run is not None:
        wandb_run.log(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-identifier",
        type=str,
        help="Huggingface Hub identifier for model.",
        default=MODEL_IDENTIFIER,
    )
    parser.add_argument(
        "--calibration-model-identifier",
        type=str,
        default=CALIBRATION_MODEL_IDENTIFIER,
        help="Huggingface Hub identifier for the calibration model.",
    )
    parser.add_argument(
        "--dataset-name", type=str, help="Name of the dataset.", choices=DATASETS
    )
    parser.add_argument(
        "--num-in-context-samples", type=int, default=NUM_IN_CONTEXT_SAMPLES
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Used batch size."
    )
    parser.add_argument(
        "--calibrator-batch-size",
        type=int,
        default=CALIBRATOR_BATCH_SIZE,
        help="Used batch size.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=WEIGHT_DECAY, help="Used weight decay"
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE, help="Used learning rate."
    )
    parser.add_argument("--input-parts", nargs="+", type=str, default=INPUT_PARTS)
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=WARMUP_FRACTION,
        help="Warm-up fraction of cosine learning rate schedule.",
    )
    parser.add_argument(
        "--use-binary-targets",
        action="store_true",
        default=False,
        help="Whether to use binary targets instead of clustering targets.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=MAX_GRAD_NORM,
        help="Maximum allowed gradient norm for gradient clipping.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=EVAL_INTERVAL,
        help="Interval during training at which we evaluate the model on the validation set.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device the model lives on.",
    )
    parser.add_argument(
        "--num-training-steps",
        type=int,
        default=NUM_TRAINING_STEPS,
        help="Number of training steps for suffix tuning.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=DATA_DIR, help="Directory containing data."
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=RESULT_DIR,
        help="Directory containing result.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed used for experiments."
    )
    parser.add_argument(
        "--track-emissions",
        action="store_true",
        default=False,
        help="Whether to track CO2eq emissions produced during the experiments.",
    )
    parser.add_argument(
        "--knock",
        action="store_true",
        default=False,
        help="Whether to announce experimental results via knockknock.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Whether to track experiments via Weights & Biases.",
    )
    parser.add_argument(
        "--notes", type=str, default=False, help="Additional notes for the experiment."
    )
    parser.add_argument(
        "--model-save-dir",
        type=str,
        default=None,
        help="Directory to save models to."
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Name of the run in Weights & Biases.",
    )

    args = parser.parse_args()

    tracker = None
    wandb_run = None

    timestamp = str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))

    if args.wandb:
        wandb_run = wandb.init(
            project=PROJECT_NAME,
            tags=[args.dataset_name, args.model_identifier],
            settings=wandb.Settings(start_method="fork"),
            config={
                "model_identifier": args.model_identifier,
                "calibration_model_identifier": args.calibration_model_identifier,
                "dataset_name": args.dataset_name,
                "num_in_context_samples": args.num_in_context_samples,
                "batch_size": args.batch_size,
                "use_binary_targets": args.use_binary_targets,
                "timestamp": timestamp,
            },
            name=args.wandb_name,
        )

    if args.track_emissions:
        timestamp = str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
        emissions_path = os.path.join(EMISSION_DIR, timestamp)
        os.makedirs(emissions_path)
        tracker = OfflineEmissionsTracker(
            project_name=PROJECT_NAME,
            country_iso_code=COUNTRY_CODE,
            output_dir=emissions_path,
            log_level="error",
        )
        tracker.start()

    # Apply decorator
    if args.knock:
        if not SECRET_IMPORTED:
            raise ImportError(
                "secret.py wasn't found, please rename secret_template.py and fill in the information."
            )

        run_single_suffix_experiment = telegram_sender(
            token=TELEGRAM_API_TOKEN, chat_id=TELEGRAM_CHAT_ID
        )(run_single_calibration_experiment)

    run_single_calibration_experiment(
        model_identifier=args.model_identifier,
        calibration_model_identifier=args.calibration_model_identifier,
        dataset_name=args.dataset_name,
        num_in_context_samples=args.num_in_context_samples,
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        calibrator_batch_size=args.calibrator_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        input_parts=args.input_parts,
        warmup_fraction=args.warmup_fraction,
        max_grad_norm=args.max_grad_norm,
        eval_interval=args.eval_interval,
        use_binary_targets=args.use_binary_targets,
        seed=args.seed,
        device=args.device,
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        model_save_dir=args.model_save_dir,
        wandb_run=wandb_run,
    )

    if tracker is not None:
        tracker.stop()

    if wandb_run is not None:
        wandb_run.finish()
