# 🍑 APRICOT: Calibrating Large Language Models Using Their Generations Only

This is the code repository for the paper [Calibrating Large Language Models Using Their Generations Only](https://aclanthology.org/2024.acl-long.824/) by 
[Dennis Ulmer](http://dennisulmer.eu/), [Martin Gubri](https://gubri.eu/), [Hwaran Lee](https://hwaranlee.github.io/), [Sangdoo Yun](https://sangdooyun.github.io/) and [Seong Joon Oh](https://coallaoh.github.io/).

Developed at [Parameter Lab](https://parameterlab.de/) with the support of [Naver AI Lab](https://clova.ai/en/research/publications.html).

## Models

The 🍑 fine-tuned models are now available on [the Hugging Face hub](https://huggingface.co/collections/parameterlab/apricot-models-673d2cae40b6ff437a86f0bf) 🤗.

Here’s how you can use them:
```python
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModel.from_pretrained("parameterlab/apricot_clustering_trivia_qa_deberta-v3-base_for_vicuna-7b-v1.5")
```

To select a different model, adjust the model name using the following pattern `apricot_{model_type}_{dataset_name}_deberta-v3-base_for_{model_name}` where:
- `{model_type}` can be either `clustering` or `binary`
- `{dataset_name}` can be either `trivia_qa` or `coqa`
- `{model_name}` can be either `vicuna-7b-v1.5` or `gpt-3.5-turbo-0125`


## Installation

The repository is simply installed by cloning the repository and installing dependencies via `pip` using Python 3.10:

    git clone https://github.com/parameterlab/apricot
    cd apricot
    pip3 install -r requirements.txt

Note that for some scripts and functionalities certain variables must be set in a `secret.py` file in the project directory or
in the form of enviroment variables. 
These include `OPENAI_API_KEY` and `OPENAI_API_KEY` when requesting data from the OpenAI API,
`WANDB_API_KEY` and `WANDB_USER_NAME` for using Weights & Biases (required for hyperparameter search), and `COUNTRY_CODE` 
for carbon emission tracking.

## Replicate results

The scripts to replicate experimental results are all given in `/experiments`.
Before running them in sequence, make sure to generate the necessary data for both datasets and models.
This can be done for TriviaQA by simply running

    python3 run_regression_experiment.py --dataset-name trivia_qa --device cuda --num-training-steps 0 --num-in-context-samples 10 --num-steps-temperature-scaling 0
    python3 get_openai_data.py --dataset-name trivia_qa

and similary for CoQA:

    python3 run_regression_experiment.py --dataset-name coqa --device cuda --num-training-steps 0 --num-in-context-samples 0 --num-steps-temperature-scaling 0
    python3 get_openai_data.py --dataset-name coqa --num-in-context-samples 0

Afterwards, run the following scripts from `/experiments`:

`sh run_main_experiments.sh` for the experimental results in section 4
`sh run_ablation_experiments.sh` and `sh run_ablation_experiments_coqa.sh` for the ablation results in appendix A.5

## Citation 

Please cite the paper as following:

```bibtex
@inproceedings{ulmer2024calibrating,
    title = "Calibrating Large Language Models Using Their Generations Only",
    author = "Ulmer, Dennis  and
      Gubri, Martin  and
      Lee, Hwaran  and
      Yun, Sangdoo  and
      Oh, Seong",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.824",
    doi = "10.18653/v1/2024.acl-long.824",
    pages = "15440--15459",
    abstract = "As large language models (LLMs) are increasingly deployed in user-facing applications, building trust and maintaining safety by accurately quantifying a model{'}s confidence in its prediction becomes even more important. However, finding effective ways to calibrate LLMs{---}especially when the only interface to the models is their generated text{---}remains a challenge. We propose APRICOT (Auxiliary prediction of confidence targets): A method to set confidence targets and train an additional model that predicts an LLM{'}s confidence based on its textual input and output alone. This approach has several advantages: It is conceptually simple, does not require access to the target model beyond its output, does not interfere with the language generation, and has a multitude of potential usages, for instance by verbalizing the predicted confidence or using it to re-prompting the LLM to accurately reflecting its uncertainty. We show how our approach performs competitively in terms of calibration error for white-box and black-box LLMs on closed-book question-answering to detect incorrect LLM answers.",
}
```

