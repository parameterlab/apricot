"""
Retrieve data used for calibration for a given split using an OpenAI model through their API.
This requires the data already generated through run_regression_experiment.py.
"""

# STD
import argparse
import os
import time
import warnings

# EXT
import dill
from httpx import HTTPStatusError
import numpy as np
from openai import OpenAI
from tqdm import tqdm

# PROJECT
from src.constants import (
    CALIBRATION_MODEL_IDENTIFIER,
    DATASETS,
    DATA_DIR,
    NUM_IN_CONTEXT_SAMPLES,
    MODEL_IDENTIFIER,
    OPENAI_MODEL_IDENTIFIER,
    DATASET_SPLIT_SIZES,
)
from src.eval import check_answer_correctness
from src.prompts import (
    QUAL_VERBALIZED_CONFIDENCE_PROMPT,
    QUANT_VERBALIZED_CONFIDENCE_PROMPT,
    QA_COT_PROMPT,
)

SECRET_IMPORTED = False
try:
    from secret import (
        OPENAI_API_KEY,
        OPENAI_ORGANIZATION_ID,
    )

    SECRET_IMPORTED = True

except ImportError:
    warnings.warn("secret.py could not be imported.")


def extract_openai_data(
    model_identifier: str,
    source_data_model_identifier: str,
    num_in_context_samples: int,
    data_dir: str,
    dataset_name: str,
):
    """
    Extract calibration data from the OpenAI API.

    Parameters
    ----------
    model_identifier: str
        Identifier of the OpenAI to use.
    source_data_model_identifier: str
        Identifier of the model whose dataloader to re-use (this would correspond to a local HF model).
    num_in_context_samples: int
        Number of in-context samples that were used with the source data model.
    data_dir: str
        Path to data directory. This is not the exact path to the dataloaders but rather the parent directory.
    dataset_name: str
        Name of the dataset to work on.
    """
    source_data_dir = os.path.join(
        data_dir,
        dataset_name,
        source_data_model_identifier.replace("/", "_"),
        "calibration_data",
        f"in_context_{num_in_context_samples}",
    )
    calibration_data_dir = os.path.join(
        data_dir,
        dataset_name,
        model_identifier.replace("/", "_"),
        "calibration_data",
        f"in_context_{num_in_context_samples}",
    )

    if not os.path.exists(calibration_data_dir):
        os.makedirs(calibration_data_dir)

    split_names = list(DATASET_SPLIT_SIZES[dataset_name].keys())

    # Load data
    if any(
        [
            not os.path.exists(
                os.path.join(source_data_dir, f"calibration_data_{split}.dill")
            )
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
                os.path.join(source_data_dir, f"calibration_data_{split}.dill"), "rb"
            ) as calibration_file:
                split_calibration_data[split] = dill.load(calibration_file)

    client = OpenAI(organization=OPENAI_ORGANIZATION_ID, api_key=OPENAI_API_KEY)

    for split in split_names:
        open_ai_calibration_data = {}

        try:
            calibration_split_path = os.path.join(
                calibration_data_dir, f"calibration_data_{split}.dill"
            )

            if os.path.exists(calibration_split_path):
                print(f"Found existing data for {split} split, skipping.")
                continue

            calibration_data = split_calibration_data[split]
            del calibration_data["included_questions"]

            for question_id, question_data in tqdm(
                calibration_data.items(), total=len(calibration_data)
            ):
                # Copy over data that is the same between models
                question = question_data["question"]
                question_in_context = question_data["question_in_context"]
                gold_answer = question_data["gold_answer"]
                open_ai_question_data = {
                    "question": question,
                    "question_in_context": question_in_context,
                    "gold_answer": gold_answer,
                    "question_embedding": question_data["question_embedding"],
                }

                # Get normal model answer
                answer_completion = client.chat.completions.create(
                    model=model_identifier,
                    messages=[{"role": "user", "content": question_in_context}],
                    logprobs=True,
                )
                answer = answer_completion.choices[0].message.content
                answer_likelihood = np.exp(
                    np.mean(
                        [
                            lp.logprob
                            for lp in answer_completion.choices[0].logprobs.content
                        ]
                    )
                )

                # Ask for verbalized uncertainty
                qual_uncertainty = (
                    client.chat.completions.create(
                        model=model_identifier,
                        messages=[
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                            {
                                "role": "user",
                                "content": QUAL_VERBALIZED_CONFIDENCE_PROMPT,
                            },
                        ],
                        max_tokens=10,
                    )
                    .choices[0]
                    .message.content
                )
                quant_uncertainty = (
                    client.chat.completions.create(
                        model=model_identifier,
                        messages=[
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                            {
                                "role": "user",
                                "content": QUANT_VERBALIZED_CONFIDENCE_PROMPT,
                            },
                        ],
                        max_tokens=10,
                    )
                    .choices[0]
                    .message.content
                )

                # Get model answer with Chain-of-though prompting
                cot_answer_completion = client.chat.completions.create(
                    model=model_identifier,
                    messages=[
                        {"role": "system", "content": QA_COT_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    logprobs=True,
                )
                cot_answer = cot_answer_completion.choices[0].message.content
                cot_answer_likelihood = np.exp(
                    np.mean(
                        [
                            lp.logprob
                            for lp in cot_answer_completion.choices[0].logprobs.content
                        ]
                    )
                )

                # Ask for verbalized uncertainty
                cot_qual_uncertainty = (
                    client.chat.completions.create(
                        model=model_identifier,
                        messages=[
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": cot_answer},
                            {
                                "role": "user",
                                "content": QUAL_VERBALIZED_CONFIDENCE_PROMPT,
                            },
                        ],
                        max_tokens=10,
                    )
                    .choices[0]
                    .message.content
                )
                cot_quant_uncertainty = (
                    client.chat.completions.create(
                        model=model_identifier,
                        messages=[
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": cot_answer},
                            {
                                "role": "user",
                                "content": QUANT_VERBALIZED_CONFIDENCE_PROMPT,
                            },
                        ],
                        max_tokens=10,
                    )
                    .choices[0]
                    .message.content
                )

                # Check correctness
                answer_correctness, cot_answer_correctness = check_answer_correctness(
                    correct_answers=[gold_answer] * 2,
                    model_answers=[answer, cot_answer],
                )

                open_ai_question_data.update(
                    {
                        "answer": answer,
                        "seq_likelihood": answer_likelihood,
                        "verbalized_qual": qual_uncertainty,
                        "verbalized_quant": quant_uncertainty,
                        "accuracy": int(answer_correctness),
                        "cot_answer": cot_answer,
                        "cot_accuracy": int(cot_answer_correctness),
                        "cot_seq_likelihood": cot_answer_likelihood,
                        "verbalized_cot_qual": cot_qual_uncertainty,
                        "verbalized_cot_quant": cot_quant_uncertainty,
                    }
                )

                # Make sure we have all the required fields
                assert question_data.keys() == open_ai_question_data.keys(), (
                    f"Some of the fields are missing for this question: "
                    f"{', '.join(list(set(question_data.keys()) - set(open_ai_question_data.keys())))}"
                )

                # Add to new dataset
                open_ai_calibration_data[question_id] = open_ai_question_data

                # Introduce a sleep phase here to avoid OpenAI rate limits
                time.sleep(0.1)

        except HTTPStatusError:
            print("API rate limit exceeded, dumping partial results.")
            calibration_split_path = os.path.join(
                calibration_data_dir, f"calibration_data_{split}_partial.dill"
            )

        finally:
            if len(open_ai_calibration_data) > 0:
                # Save the calibration data
                with open(calibration_split_path, "wb") as calibration_file:
                    dill.dump(open_ai_calibration_data, calibration_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-identifier",
        type=str,
        default=OPENAI_MODEL_IDENTIFIER,
        help="OpenAI identifier for model.",
    )
    parser.add_argument(
        "--calibration-model-identifier",
        type=str,
        default=CALIBRATION_MODEL_IDENTIFIER,
        help="Identifier of the Huggingface model used for calibration purposes.",
    )
    parser.add_argument(
        "--source-data-model-identifier",
        type=str,
        default=MODEL_IDENTIFIER,
        help="Identifier of the Huggingface model for which the data was originally preprocessed for.",
    )

    parser.add_argument(
        "--dataset-name", type=str, help="Name of the dataset.", choices=DATASETS
    )
    parser.add_argument(
        "--num-in-context-samples", type=int, default=NUM_IN_CONTEXT_SAMPLES
    )
    parser.add_argument(
        "--data-dir", type=str, default=DATA_DIR, help="Directory containing data."
    )

    args = parser.parse_args()

    extract_openai_data(
        model_identifier=args.model_identifier,
        source_data_model_identifier=args.source_data_model_identifier,
        dataset_name=args.dataset_name,
        num_in_context_samples=args.num_in_context_samples,
        data_dir=args.data_dir,
    )
