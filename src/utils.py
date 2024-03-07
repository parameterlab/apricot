"""
Define some additional utility functions.
"""

# STD
from typing import Tuple, List, Dict, Any

# EXT
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


def unpack_dataloader(
    dataloader: DataLoader, tokenizer: AutoTokenizer
) -> Tuple[List[str], List[str]]:
    """
    Unpack an existing dataloader into two lists of question ids and model inputs.

    Parameters
    ----------
    dataloader: DataLoader
        Dataloader to be unpacked.
    tokenizer: AutoTokenizer
        Tokenizer used to decode model inputs.

    Returns
    -------
    Tuple[List[str], List[str]]
        Two lists of question ids and model inputs.
    """
    inputs_, question_ids = [], []

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"]
        if len(input_ids.shape) == 3:
            input_ids = input_ids.squeeze(1)
        inputs_ += tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        question_ids += batch["question_id"]

    return inputs_, question_ids


def loop_dataloader(dataloader: DataLoader):
    """
    Loop through a dataloader infinitely.

    Parameters
    ----------
    dataloader: Dataloder
        Dataloader to be looped through.

    Yields
    ------
    batch: Dict[str, Any]
        Batch from dataloader.
    """
    while True:
        for batch in dataloader:
            yield batch


def create_calibration_dataloader(
    batch_size: int,
    inputs_: List[str],
    question_ids: List[str],
    calibration_targets: Dict[str, float],
    calibration_data: Dict[str, Dict[str, Any]],
    tokenizer: AutoTokenizer,
    **tokenizer_kwargs,
) -> DataLoader:
    """
    Create dataloader for the calibration model.

    Parameters
    ----------
    batch_size: int
        Batch size for the calibration model.
    inputs_: List[str]
        List of inputs for the calibration model.
    question_ids: List[str]
        Question ids corresponding to the inputs.
    calibration_targets: Dict[str, float]
        Dictionary mapping from question ids to calibration targets.
    calibration_data: Dict[str, Dict[str, Any]]
        Dictionary mapping from question id to all the corresponding data extracted from the target LLM.
    tokenizer: AutoTokenizer
        Tokenizer for the calibration model.

    Returns
    -------
    DataLoader
        Created dataloader.
    """
    data = []

    for input_, question_id in zip(inputs_, question_ids):
        tokenized_input = tokenizer(input_, **tokenizer_kwargs)
        target = calibration_targets[question_id]
        data.append(
            {
                **tokenized_input,
                "target": target,
                "question_id": question_id,
                "correctness": calibration_data[question_id]["accuracy"],
            }
        )

    data_loader = DataLoader(data, batch_size=batch_size)

    return data_loader
