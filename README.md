# 🍑 APRICOT: Calibrating Large Language Models Using Their Generations Only

This is the code repository for the paper [Calibrating Large Language Models Using Their Generations Only]() by 
[Dennis Ulmer](http://dennisulmer.eu/), [Martin Gubri](https://gubri.eu/), [Hwaran Lee](https://hwaranlee.github.io/), [Sangdoo Yun](https://sangdooyun.github.io/) and [Seong Joon Oh](https://coallaoh.github.io/).
Developed at [Parameter Lab](https://parameterlab.de/) with the support of [Naver AI Lab](https://clova.ai/en/research/publications.html).

## Installation

The repository is simply installed by cloning the repository and installing dependencies via `pip`:

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

[Citation coming soon]
