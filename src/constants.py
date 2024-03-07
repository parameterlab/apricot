"""
Common module to define default values and configurations.
"""

# DEFAULT PATHS
PROJECT_NAME = "apricot"
CACHE_DIR = "/tmp"
DATA_DIR = "./data"
RESULT_DIR = "./results"
EMISSION_DIR = "./emissions"

# MODEL IDENTIFIERS
MODEL_IDENTIFIER = "lmsys/vicuna-7b-v1.5"
OPENAI_MODEL_IDENTIFIER = "gpt-3.5-turbo-0125"
CALIBRATION_MODEL_IDENTIFIER = "microsoft/deberta-v3-base"
SENTENCE_EMBEDDING_MODEL_IDENTIFIER = "all-mpnet-base-v2"
BLACK_BOX_MODELS = ["gpt-3.5-turbo-0125"]

# LLM HYPERPARAMETERS
BATCH_SIZE = 4
NUM_IN_CONTEXT_SAMPLES = 10
END_OF_GENERATION_TOKENS = [
    "Question:",
    " Question:",
    "Question: ",
    "\n",
    "Answer:",
    "\nQuestion:",
    " Answer:",
    "Q:",
]
QUALITATIVE_SCALE = {
    "Very low": 0,
    "Low": 0.3,
    "Somewhat low": 0.45,
    "Medium": 0.5,
    "Somewhat high": 0.65,
    "High": 0.7,
    "Very high": 1,
}


# CALIBRATOR HYPERPARAMETERS
CALIBRATOR_BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.1
WARMUP_FRACTION = 0.1
NUM_TRAINING_STEPS = 600
MAX_INPUT_LENGTH = 512
MAX_GRAD_NORM = 10
EVAL_INTERVAL = 50
SEED = 1234
INPUT_PARTS = ["question", "answer"]  # Default composition of inputs
ALLOWED_INPUTS = {
    "question",
    "answer",
    "cot_answer",
    "qualitative",
    "quantitative",
}  # Allowed parts for input
CALIBRATION_MODEL_PARAMS = {
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "max_grad_norm": MAX_GRAD_NORM,
    "warmup_fraction": WARMUP_FRACTION,
    "num_training_steps": NUM_TRAINING_STEPS,
    "eval_interval": EVAL_INTERVAL,
}

# PLATT SCALING HYPERPARAMETERS
PLATT_SCALING_BATCH_SIZE = 64
PLATT_SCALING_LEARNING_RATE = 0.01
PLATT_SCALING_NUM_STEPS = 200
PLATT_SCALING_VALID_INTERVAL = 20

# DATASET DETAILS
DATASETS = ("trivia_qa", "coqa")
DATASET_SPLIT_SIZES = {
    "trivia_qa": {"train": 12000, "valid": 1500, "test": 1500},
    "coqa": {"train": 10000, "valid": 1500, "test": 1500},
}

# BASELINES & EVALUATIONS
BASELINES_METHODS = [
    "seq_likelihood",
    "cot_seq_likelihood",
    "ts_seq_likelihood",
    "ts_cot_seq_likelihood",
    "qual_verbalized_uncertainty",
    "cot_qual_verbalized_uncertainty",
    "quant_verbalized_uncertainty",
    "cot_quant_verbalized_uncertainty",
]
EVAL_METRIC_ORDER = ["brier_score", "ece", "smece", "auroc"]
