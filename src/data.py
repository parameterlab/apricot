"""
Module containing any data-related utlities and functions.
"""

# STD
from collections import namedtuple
import hashlib
import os
from typing import Dict, Any, Callable

# EXT
import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# PROJECT
from src.constants import DATASETS, MAX_INPUT_LENGTH
from src.prompts import QA_FEW_SHOT_TEMPLATE, QA_COT_PROMPT, QA_OPEN_BOOK_TEMPLATE

# CUSTOM
DataSplits = namedtuple("DataSplits", ["validation", "test"])


def process_batch_wrapper(
    train_data: datasets.Dataset,
    num_in_context_samples: int,
    tokenizer: AutoTokenizer,
    max_input_length: int,
) -> Callable:
    """
    Closure for the process batch function that makes a certain variables available to the function scope.
    """

    def process_batch(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a specific batch of TriviaQA.

        Parameters
        ----------
        batch: Dict[str, Any]
            Batch of inputs of the dataset.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of torch tensors, containing the important data for training.
        """
        answer_field = "answer" if "answer" in batch else "best_answer"

        answers = [
            str(answer) if not isinstance(answer, dict) else answer["value"]
            for answer in batch[answer_field]
        ]

        # Select few shot examples
        few_shot_prompts = []

        for _ in range(len(answers)):
            few_shot_prompt = ""

            if num_in_context_samples > 0:
                train_indices = np.random.choice(
                    range(0, len(train_data)), size=num_in_context_samples
                )
                in_context_samples = train_data.select(train_indices)

                for sample in in_context_samples:
                    answer = (
                        sample["answer"]
                        if "answer" in sample and not isinstance(sample["answer"], dict)
                        else sample.get("best_answer", sample["answer"]["value"])
                    )
                    few_shot_prompt += QA_FEW_SHOT_TEMPLATE.format(
                        question=sample["question"], answer=answer
                    )

            few_shot_prompts.append(few_shot_prompt)

        batch_with_prompt = [
            few_shot_prompt + "Question: " + question + " Answer:"
            for question, few_shot_prompt in zip(batch["question"], few_shot_prompts)
        ]
        batch_with_cot_prompt = [
            few_shot_prompt + QA_COT_PROMPT + " Question: " + question + " Answer:"
            for question, few_shot_prompt in zip(batch["question"], few_shot_prompts)
        ]
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            batch_with_prompt,
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
        )
        cot_inputs = tokenizer(
            batch_with_cot_prompt,
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["cot_input_ids"] = cot_inputs.input_ids
        batch["cot_attention_mask"] = cot_inputs.attention_mask
        batch["answer"] = answers

        # Generate question IDs for OOD test set
        if "question_id" not in batch:
            batch["question_id"] = []
            h = hashlib.new("sha256")

            for question in batch["question"]:
                h.update(question.encode())
                batch["question_id"].append(h.hexdigest())

        return batch

    return process_batch


def preprocess_trivia_qa(
    model_identifier: str,
    num_in_context_samples: int,
    batch_size: int,
    data_dir: str,
    validation_fraction: float = 0.01,
    max_input_length: int = MAX_INPUT_LENGTH,
) -> Dict[str, DataLoader]:
    """
    Preprocess the TriviaQA dataset. This involves preparing the inputs by adding a number in-context samples and using
    the target model's tokenizer. This function is loosely based on the code by Lorenz Kuhn:
    https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_triviaqa.py.

    Parameters
    ----------
    model_identifier: str
        Identifier of the target model.
    num_in_context_samples: int
        Number of in-context learning samples to include in the prompt.
    batch_size: int
        Batch size.
    data_dir: str
        Directory where to save data to. More precisely, we save the data in the "processed/trivia_qa" subfolder.
    validation_fraction: float
        Fraction of samples to use from the training set to use as the validation set.
    max_input_length: int
        Maximum length for input (with in-context examples). Input will be padded up to that length.

    Returns
    -------
    Dict[str, DataLoader]
        Dataloaders of the validation and the test set as a dictionary.
    """
    processed_data_dir = os.path.join(
        data_dir,
        "trivia_qa",
        model_identifier.replace("/", "_"),
        "processed",
        f"in_context_{num_in_context_samples}",
    )

    train_data_loader_path = os.path.join(processed_data_dir, "train.dl")
    validation_data_loader_path = os.path.join(processed_data_dir, "validation.dl")
    test_data_loader_path = os.path.join(processed_data_dir, "test.dl")

    if (
        not os.path.exists(train_data_loader_path)
        or not os.path.exists(validation_data_loader_path)
        or not os.path.exists(test_data_loader_path)
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_identifier)

        # Load data splits - test split does not contain answers, so use (part of) training data for calibration and
        # validation data for testing
        formatted_percentage = str(int(100 - validation_fraction * 100))
        train_data = datasets.load_dataset(
            "trivia_qa", "rc.nocontext", split=f"train[:{formatted_percentage}%]"
        )
        val_data = datasets.load_dataset(
            "trivia_qa", "rc.nocontext", split=f"train[{formatted_percentage}%:]"
        )
        test_data = datasets.load_dataset(
            "trivia_qa", "rc.nocontext", split="validation"
        )

        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)

        for split_name, data_split, data_loader_path in zip(
            ["train", "validation", "test"],
            [train_data, val_data, test_data],
            [
                train_data_loader_path,
                validation_data_loader_path,
                test_data_loader_path,
            ],
        ):
            if os.path.exists(data_loader_path):
                continue

            remove_columns = ["search_results", "question_source", "entity_pages"]

            if split_name == "ood_test":
                remove_columns = ["type", "source"]

            data_split = data_split.map(
                # Create processing function by making variables available in this closure
                process_batch_wrapper(
                    train_data=train_data,
                    num_in_context_samples=num_in_context_samples,
                    tokenizer=tokenizer,
                    max_input_length=max_input_length,
                ),
                batched=True,
                batch_size=batch_size,
                remove_columns=remove_columns,
            )
            data_split.set_format(
                type="torch",
                columns=[
                    "input_ids",
                    "attention_mask",
                    "cot_input_ids",
                    "cot_attention_mask",
                ],
                output_all_columns=True,
            )
            data_loader = DataLoader(data_split, batch_size=batch_size, drop_last=True)

            data_split.save_to_disk(
                os.path.join(processed_data_dir, f"{split_name}.data")
            )
            torch.save(
                data_loader, os.path.join(processed_data_dir, f"{split_name}.dl")
            )

    data_loaders = {
        "train": torch.load(train_data_loader_path),
        "valid": torch.load(validation_data_loader_path),
        "test": torch.load(test_data_loader_path),
    }

    return data_loaders


def preprocess_coqa(
    model_identifier: str,
    num_in_context_samples: int,
    batch_size: int,
    data_dir: str,
    max_input_length: int = MAX_INPUT_LENGTH,
) -> Dict[str, DataLoader]:
    """
    Preprocess the CoQA dataset. This involves preparing the inputs
    and using the target model's tokenizer. This function is loosely based on the code by Lorenz Kuhn:
    https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_triviaqa.py.

    Parameters
    ----------
    model_identifier: str
        Identifier of the target model.
    num_in_context_samples: int
        Number of in-context learning samples to include in the prompt.
    batch_size: int
        Batch size.
    data_dir: str
        Directory where to save data to. More precisely, we save the data in the "processed/coqa" subfolder.
    max_input_length: int
        Maximum length for input (with in-context examples). Input will be padded up to that length.

    Returns
    -------
    Dict[str, DataLoader]
        Dataloaders of the validation and the test set as a dictionary.
    """
    assert num_in_context_samples == 0, "Few-shot learning not supported for CoQA."

    processed_data_dir = os.path.join(
        data_dir,
        "coqa",
        model_identifier.replace("/", "_"),
        "processed",
        f"in_context_{num_in_context_samples}",
    )

    train_data_loader_path = os.path.join(processed_data_dir, "train.dl")
    validation_data_loader_path = os.path.join(processed_data_dir, "validation.dl")
    test_data_loader_path = os.path.join(processed_data_dir, "test.dl")

    tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    tokenizer.padding_side = "left"

    if any(
        [
            not os.path.exists(data_loader_path)
            for data_loader_path in [
                train_data_loader_path,
                validation_data_loader_path,
                test_data_loader_path,
            ]
        ]
    ):
        # Load data splits
        train_data = datasets.load_dataset("stanfordnlp/coqa", split="train")
        val_data = datasets.load_dataset("stanfordnlp/coqa", split="validation[:50%]")
        test_data = datasets.load_dataset("stanfordnlp/coqa", split="validation[50%:]")

        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)

        for split_name, data_split, data_loader_path in zip(
            ["train", "validation", "test"],
            [train_data, val_data, test_data],
            [
                train_data_loader_path,
                validation_data_loader_path,
                test_data_loader_path,
            ],
        ):
            if os.path.exists(data_loader_path):
                continue

            split_samples = []
            for sample in tqdm(data_split, total=len(data_split)):
                for question, answer in zip(
                    sample["questions"], sample["answers"]["input_text"]
                ):
                    h = hashlib.new("sha256")
                    h.update(f"{question}{answer}".encode())

                    prompt = QA_OPEN_BOOK_TEMPLATE.format(
                        context=sample["story"], cot_prompt="", question=question
                    )
                    cot_prompt = QA_OPEN_BOOK_TEMPLATE.format(
                        context=sample["story"],
                        cot_prompt=f"Instruction: {QA_COT_PROMPT}\n",
                        question=question,
                    )
                    inputs = tokenizer(
                        prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=max_input_length,
                        return_tensors="pt",
                    )
                    cot_inputs = tokenizer(
                        cot_prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=max_input_length,
                        return_tensors="pt",
                    )

                    question_data = {
                        "question_id": h.hexdigest(),
                        "input_ids": inputs.input_ids.squeeze(0),
                        "attention_mask": inputs.attention_mask.squeeze(0),
                        "cot_input_ids": cot_inputs.input_ids.squeeze(0),
                        "cot_attention_mask": cot_inputs.attention_mask.squeeze(0),
                        "question": question,
                        "prompt": prompt,
                        "cot_prompt": cot_prompt,
                        "answer": answer,
                    }
                    split_samples.append(question_data)

            data_loader = DataLoader(split_samples, batch_size=batch_size)

            data_split.save_to_disk(
                os.path.join(processed_data_dir, f"{split_name}.data")
            )
            torch.save(
                data_loader, os.path.join(processed_data_dir, f"{split_name}.dl")
            )

    data_loaders = {
        "train": torch.load(train_data_loader_path),
        "valid": torch.load(validation_data_loader_path),
        "test": torch.load(test_data_loader_path),
    }

    return data_loaders


def preprocess_dataset(
    dataset_name: str,
    model_identifier: str,
    num_in_context_samples: int,
    batch_size: int,
    data_dir: str,
) -> Dict[str, DataLoader]:
    """
    Preprocess one of the datasets used in this project.

    Parameters
    ----------
    dataset_name: str
        Name of the target dataset.
    model_identifier: str
        Identifier of the target model.
    num_in_context_samples: int
        Number of in-context learning samples to include in the prompt.
    batch_size: int
        Batch size.
    data_dir: str
        Directory containing datasets.

    Returns
    -------
    Dict[str, DataLoader]
        Dataloaders of the validation and the test set.
    """
    assert (
        dataset_name in DATASETS
    ), f"dataset_name must be one of {' ,'.join(DATASETS)}, '{dataset_name}' found instead."

    if dataset_name == "trivia_qa":
        return preprocess_trivia_qa(
            model_identifier=model_identifier,
            num_in_context_samples=num_in_context_samples,
            batch_size=batch_size,
            data_dir=data_dir,
        )

    elif dataset_name == "coqa":
        return preprocess_coqa(
            model_identifier=model_identifier,
            num_in_context_samples=num_in_context_samples,
            batch_size=batch_size,
            data_dir=data_dir,
        )

    else:
        raise NotImplementedError(
            f"Dataset {dataset_name} not supported. Please add custom code for preprocessing."
        )
