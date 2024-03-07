"""
Collect functions that are concerned with assessing or improving the calibration of the model.
"""

# STD
import os
from typing import Dict, Any, Optional, List, Tuple

# EXT
import dill
import numpy as np
from optimum.bettertransformer import BetterTransformer
from pacmap import PaCMAP
from sentence_transformers import SentenceTransformer
import sklearn.cluster as cluster
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# PROJECT
from src.constants import (
    END_OF_GENERATION_TOKENS,
    MAX_INPUT_LENGTH,
    SENTENCE_EMBEDDING_MODEL_IDENTIFIER,
)
from src.eval import check_answer_correctness
from src.prompts import (
    QUAL_VERBALIZED_CONFIDENCE_PROMPT,
    QUANT_VERBALIZED_CONFIDENCE_PROMPT,
)


def extract_model_calibration_data(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    calibration_split: DataLoader,
    device: str,
    sentence_embedding_model_identifier: str = SENTENCE_EMBEDDING_MODEL_IDENTIFIER,
    max_generation_length: int = 50,
    max_samples: Optional[int] = None,
    max_input_length: int = MAX_INPUT_LENGTH,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Extract data from the target LLM that is being used for calibration purposes.
    This includes:
        * The model answer.
        * The sequence likelihood of the model answer.
        * The quantitative verbalized uncertainty, i.e. its confidence in percent.
        * The qualitative verbalized uncertainty, i.e. its confidence expressed on a verbal scale.
        * All of the above, after using chain-of-thought prompting.
        * Sentence embedding of the input question.

    Parameters
    ----------
    model: AutoModelForCausalLM
        Model to be used for experiments.
    tokenizer: AutoTokenizer
        Tokenizer to be used for experiments.
    calibration_split: DataLoader
        Dataloader used for calibration.
    device: str
        Device the model lives on.
    sentence_embedding_model_identifier: str
        Identifier of the sentence embedding model. Defaults to SENTENCE_EMBEDDING_MODEL_IDENTIFIER defined in
        src.constants.
    max_generation_length: int
        Maximum length for the generation answer.
    max_samples: Optional[int]
        Maximum number of samples to process from the current split. Default is None, in which case we process all
        available samples.
    max_input_length: int
        Maximum input length. Default is MAX_INPUT_LENGTH defined in src.constants.

    Returns
    -------
    Tuple[Dict[str, Dict[str, Any]], List[str]]
        All the relevant data for calibration extracted from the model, including its answer, correctness and
        confidence. Additionally, return a list of all question ids that have been processed.
    """
    calibration_data = {}
    eos_token_ids = [
        [tokenizer(eos_token)["input_ids"][1]] for eos_token in END_OF_GENERATION_TOKENS
    ]
    try:
        model = BetterTransformer.transform(
            model
        )  # Necessary for flash attention to work

    except Exception:
        # When model is already using BetterTransformer
        pass

    embedding_model = SentenceTransformer(sentence_embedding_model_identifier).to(
        device
    )

    if max_samples is None:
        max_samples = len(calibration_split)
        total = len(calibration_split) // calibration_split.batch_size

    else:
        total = max_samples // calibration_split.batch_size

    num_current_samples = 0
    included_questions = []
    for i, batch in tqdm(enumerate(calibration_split), total=total):
        num_current_samples += batch["input_ids"].shape[0]

        if num_current_samples > max_samples:
            break

        inputs = batch["input_ids"].to(device)
        questions_in_context = tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        attention_mask = batch["attention_mask"].to(device)
        cot_inputs = batch["cot_input_ids"].to(device)
        cot_attention_mask = batch["cot_attention_mask"].to(device)

        with torch.no_grad() and torch.backends.cuda.sdp_kernel(
            enable_flash=True  # Enable flash attention
        ):
            outputs = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_length=inputs.shape[1] + max_generation_length,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=eos_token_ids,
            )

            cot_outputs = model.generate(
                input_ids=cot_inputs,
                attention_mask=cot_attention_mask,
                max_length=cot_inputs.shape[1] + max_generation_length,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=eos_token_ids,
            )

        # Get sequence likelihoods
        generated_answer_ids = outputs["sequences"][:, inputs.shape[1] :].squeeze(0)
        predictions = torch.log(
            F.softmax(torch.stack(outputs["scores"], dim=1), dim=-1)
        )
        log_probs = torch.gather(
            predictions, dim=-1, index=generated_answer_ids.unsqueeze(-1)
        ).squeeze(-1)
        token_mask = torch.all(
            torch.stack(
                [
                    generated_answer_ids != token_id
                    for token_id in tokenizer.all_special_ids
                ],
                dim=-1,
            ),
            dim=-1,
        ).long()
        num_tokens = token_mask.sum(dim=-1)
        seq_likelihoods = (log_probs * token_mask).sum(-1) / num_tokens
        seq_likelihoods = torch.exp(seq_likelihoods)

        cot_generated_answer_ids = cot_outputs["sequences"][
            :, cot_inputs.shape[1] :
        ].squeeze(0)
        cot_predictions = torch.log(
            F.softmax(torch.stack(cot_outputs["scores"], dim=1), dim=-1)
        )
        cot_log_probs = torch.gather(
            cot_predictions, dim=-1, index=cot_generated_answer_ids.unsqueeze(-1)
        ).squeeze(-1)
        cot_token_mask = torch.all(
            torch.stack(
                [
                    cot_generated_answer_ids != token_id
                    for token_id in tokenizer.all_special_ids
                ],
                dim=-1,
            ),
            dim=-1,
        ).long()
        cot_num_tokens = cot_token_mask.sum(dim=-1)
        cot_seq_likelihoods = (cot_log_probs * cot_token_mask).sum(-1) / cot_num_tokens
        cot_seq_likelihoods = torch.exp(cot_seq_likelihoods)

        # Decode output to answer
        generated_answer_ids = outputs["sequences"][:, inputs.shape[1] :].squeeze(0)
        model_answers = tokenizer.batch_decode(
            generated_answer_ids, skip_special_tokens=True
        )

        cot_generated_answer_ids = cot_outputs["sequences"][
            :, cot_inputs.shape[1] :
        ].squeeze(0)
        cot_model_answers = tokenizer.batch_decode(
            cot_generated_answer_ids, skip_special_tokens=True
        )

        # Check correctness
        answers_correctness = check_answer_correctness(
            correct_answers=batch["answer"],
            model_answers=model_answers,
        )
        cot_answers_correctness = check_answer_correctness(
            correct_answers=batch["answer"], model_answers=cot_model_answers
        )

        # Retrieve verbalized uncertainty
        # Qualitative verbalized - use words such as "very low" or "high"
        raw_qual_inputs = [
            f"{question} {answer} {QUAL_VERBALIZED_CONFIDENCE_PROMPT}"
            for answer, question in zip(model_answers, batch["question"])
        ]
        raw_cot_qal_inputs = [
            f"{question} {answer} {QUAL_VERBALIZED_CONFIDENCE_PROMPT}"
            for answer, question in zip(cot_model_answers, batch["question"])
        ]
        qual_inputs = tokenizer(
            raw_qual_inputs,
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )
        cot_qual_inputs = tokenizer(
            raw_cot_qal_inputs,
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )

        # Quantitive - use percentage values
        raw_quant_inputs = [
            f"{question} {answer} {QUANT_VERBALIZED_CONFIDENCE_PROMPT}"
            for answer, question in zip(model_answers, batch["question"])
        ]
        raw_cot_qant_inputs = [
            f"{question} {answer} {QUANT_VERBALIZED_CONFIDENCE_PROMPT}"
            for answer, question in zip(cot_model_answers, batch["question"])
        ]
        quant_inputs = tokenizer(
            raw_quant_inputs,
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )
        cot_quant_inputs = tokenizer(
            raw_cot_qant_inputs,
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )

        verbalized_uncertainties = {}

        with torch.no_grad() and torch.backends.cuda.sdp_kernel(
            enable_flash=True  # Enable flash attention
        ):
            for name, tokenized_inputs in zip(
                ["qual", "cot_qual", "quant", "cot_quant"],
                [qual_inputs, cot_qual_inputs, quant_inputs, cot_quant_inputs],
            ):
                inputs = tokenized_inputs["input_ids"].to(device)
                attention_mask = tokenized_inputs["attention_mask"].to(device)
                outputs = model.generate(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    max_length=inputs.shape[1] + 20,
                    return_dict_in_generate=True,
                    eos_token_id=tokenizer.eos_token_id,
                    bad_words_ids=eos_token_ids,
                )
                generated_answer_ids = outputs["sequences"][
                    :, inputs.shape[1] :
                ].squeeze(0)
                verbalized_uncertainties[name] = tokenizer.batch_decode(
                    generated_answer_ids, skip_special_tokens=True
                )

        # Initial sentence embedding model
        included_questions += batch["question_id"]
        question_embeddings = embedding_model.encode(batch["question"])

        # Create the calibration data using this mega for-loop
        for (
            question_id,
            question,
            question_in_context,
            model_answer,
            gold_answer,
            correctness,
            cot_model_answer,
            cot_correctness,
            verbalized_quant,
            verbalized_qual,
            verbalized_cot_quant,
            verbalized_cot_qual,
            seq_likelihood,
            cot_seq_likelihood,
            question_embedding,
        ) in zip(
            batch["question_id"],
            batch["question"],
            questions_in_context,
            model_answers,
            batch["answer"],
            answers_correctness,
            cot_model_answers,
            cot_answers_correctness,
            verbalized_uncertainties["quant"],
            verbalized_uncertainties["qual"],
            verbalized_uncertainties["cot_quant"],
            verbalized_uncertainties["cot_qual"],
            seq_likelihoods,
            cot_seq_likelihoods,
            question_embeddings,
        ):
            calibration_data[question_id] = {
                "accuracy": int(correctness),
                "cot_accuracy": int(cot_correctness),
                "gold_answer": gold_answer,
                "answer": model_answer,
                "cot_answer": cot_model_answer,
                "question": question,
                "question_in_context": question_in_context,
                "question_embedding": question_embedding,
                "verbalized_quant": verbalized_quant,
                "verbalized_qual": verbalized_qual,
                "verbalized_cot_quant": verbalized_cot_quant,
                "verbalized_cot_qual": verbalized_cot_qual,
                "seq_likelihood": seq_likelihood.cpu().item(),
                "cot_seq_likelihood": cot_seq_likelihood.cpu().item(),
            }

        del outputs

    return calibration_data, included_questions


def compute_question_calibration_targets(
    calibration_data: Dict[str, Dict[str, float]],
    data_dir: Optional[str] = None,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute calibration targets: This is done by clustering question embeddings in latent space, and computing the
    observed accuracy per cluster.

    Parameters
    ----------
    calibration_data: Dict[str, Dict[str, float]]
        Calibration data used to compute calibration targets.
    data_dir: Optional[str]
        Data directory to store clustering results to if not None.
    eps: float
        Small value to add to embedding normalization to avoid nan values.

    Returns
    -------
    Dict[str, float]
        Calibration targets mapping from question IDs to calibration target values.
    """
    question_ids, questions, accuracies, embeddings = [], [], [], []

    for question_id, question_data in tqdm(calibration_data.items()):
        accuracies.append(question_data["accuracy"])
        question_ids.append(question_id)
        embeddings.append(question_data["question_embedding"])
        questions.append(question_data["question"])

    accuracies = np.array(accuracies)
    embeddings = np.stack(embeddings, axis=0)
    questions = np.array(questions)

    dbscan = cluster.HDBSCAN(
        min_cluster_size=3, min_samples=1, n_jobs=1
    )  # Set min_samples 1 to make sure all questions are clustered
    embeddings = (embeddings - np.mean(embeddings, axis=0)) / (
        np.std(embeddings, axis=0) + eps
    )
    embeddings = embeddings.astype(np.float16)  # To not blow up system memory
    dbscan.fit(embeddings)
    cluster_labels = dbscan.labels_
    label2target = {
        label: np.mean(accuracies[cluster_labels == label])
        for label in set(cluster_labels)
    }

    # Plot results
    if data_dir is not None:
        # Perform dimensionality reduction
        pacmap = PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
        transformed_embeddings = pacmap.fit_transform(embeddings, init="pca")

        with open(
            os.path.join(data_dir, "cluster_contents.txt"), "w"
        ) as cluster_token_file:
            for label in set(cluster_labels):
                cluster_token_file.write(f"\n#### Cluster {label} ####\n")
                cluster_token_file.write("\n".join(questions[cluster_labels == label]))

        cluster_data = {
            "accuracies": accuracies,
            "embeddings": embeddings,
            "questions": questions,
            "transformed_embeddings": transformed_embeddings,
            "cluster_assignments": cluster_labels,
            "calibration_targets": label2target,
        }

        with open(
            os.path.join(data_dir, "cluster_data.dill"), "wb"
        ) as cluster_data_file:
            dill.dump(cluster_data, cluster_data_file)

    # Map question IDs to target values
    question_id_to_targets = {
        question_id: label2target[cluster_labels[i]]
        if cluster_labels[i] > 0
        else accuracies[i]
        for i, question_id in enumerate(question_ids)
    }

    return question_id_to_targets
