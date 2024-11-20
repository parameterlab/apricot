cd ..
result_dir="./results_ablations"
data_dir="./data"

# Run Vicuna experiments / CoQA
python3 run_regression_experiment.py --model-identifier lmsys/vicuna-7b-v1.5 --dataset-name coqa --device cuda:0 --num-training-steps 600 --num-in-context-samples 0 --data-dir /data --result-dir $result_dir --lr 0.00008836 --weight-decay 0.0007421 --input-parts question
python3 run_regression_experiment.py --model-identifier lmsys/vicuna-7b-v1.5 --dataset-name coqa --device cuda:0 --num-training-steps 600 --num-in-context-samples 0 --data-dir /data --result-dir $result_dir --lr 0.00008836 --weight-decay 0.0007421 --input-parts question answer qualitative
python3 run_regression_experiment.py --model-identifier lmsys/vicuna-7b-v1.5 --dataset-name coqa --device cuda:0 --num-training-steps 600 --num-in-context-samples 0 --data-dir /data --result-dir $result_dir --lr 0.00008836 --weight-decay 0.0007421 --input-parts question answer quantitative
python3 run_regression_experiment.py --model-identifier lmsys/vicuna-7b-v1.5 --dataset-name coqa --device cuda:0 --num-training-steps 600 --num-in-context-samples 0 --data-dir /data --result-dir $result_dir --lr 0.00008836 --weight-decay 0.0007421 --input-parts question cot_answer
python3 run_regression_experiment.py --model-identifier lmsys/vicuna-7b-v1.5 --dataset-name coqa --device cuda:0 --num-training-steps 600 --num-in-context-samples 0 --data-dir /data --result-dir $result_dir --lr 0.00008836 --weight-decay 0.0007421 --input-parts question cot_answer qualitative
python3 run_regression_experiment.py --model-identifier lmsys/vicuna-7b-v1.5 --dataset-name coqa --device cuda:0 --num-training-steps 600 --num-in-context-samples 0 --data-dir /data --result-dir $result_dir --lr 0.00008836 --weight-decay 0.0007421 --input-parts question cot_answer quantitative

# Run GPT-3.5 experiments / CoQA
python3 run_regression_experiment.py --model-identifier gpt-3.5-turbo-0125 --dataset-name coqa --device cuda:0 --num-training-steps 600 --num-in-context-samples 0 --data-dir /data --result-dir $result_dir --lr 0.00005592 --weight-decay 0.03495 --input-parts question
python3 run_regression_experiment.py --model-identifier gpt-3.5-turbo-0125 --dataset-name coqa --device cuda:0 --num-training-steps 600 --num-in-context-samples 0 --data-dir /data --result-dir $result_dir --lr 0.00005592 --weight-decay 0.03495 --input-parts question answer qualitative
python3 run_regression_experiment.py --model-identifier gpt-3.5-turbo-0125 --dataset-name coqa --device cuda:0 --num-training-steps 600 --num-in-context-samples 0 --data-dir /data --result-dir $result_dir --lr 0.00005592 --weight-decay 0.03495 --input-parts question answer quantitative
python3 run_regression_experiment.py --model-identifier gpt-3.5-turbo-0125 --dataset-name coqa --device cuda:0 --num-training-steps 600 --num-in-context-samples 0 --data-dir /data --result-dir $result_dir --lr 0.00005592 --weight-decay 0.03495 --input-parts question cot_answer
python3 run_regression_experiment.py --model-identifier gpt-3.5-turbo-0125 --dataset-name coqa --device cuda:0 --num-training-steps 600 --num-in-context-samples 0 --data-dir /data --result-dir $result_dir --lr 0.00005592 --weight-decay 0.03495 --input-parts question cot_answer qualitative
python3 run_regression_experiment.py --model-identifier gpt-3.5-turbo-0125 --dataset-name coqa --device cuda:0 --num-training-steps 600 --num-in-context-samples 0 --data-dir /data --result-dir $result_dir --lr 0.00005592 --weight-decay 0.03495 --input-parts question cot_answer quantitative