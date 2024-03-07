"""
Define all the prompts used in this project in a single module.
"""

# PROJECT
from src.constants import QUALITATIVE_SCALE

QA_FEW_SHOT_TEMPLATE = "Question: {question} Answer: {answer} "
QA_OPEN_BOOK_TEMPLATE = (
    "Context: {context}\n\n{cot_prompt}Question: {question}\nAnswer: "
)
QA_COT_PROMPT = "Briefly answer the following question by thinking step by step."
QUANT_VERBALIZED_CONFIDENCE_PROMPT = (
    ".\nPlease provide your confidence in the answer only in percent (0-100 %): "
)
QUAL_VERBALIZED_CONFIDENCE_PROMPT = f".\nPlease provide your confidence in the answer only as one of {' / '.join([f'{exp}' for exp in QUALITATIVE_SCALE.keys()])}: "
