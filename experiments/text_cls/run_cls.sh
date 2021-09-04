export TASK_NAME=sst2
export MODEL_NAME=textattack/bert-base-uncased-SST-2

python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 1 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/

python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 0.95 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/

python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 0.9 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/

python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 0.85 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/

python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 0.8 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/

python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 0.75 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/


python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 0.7 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/

python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 0.65 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/


python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 0.6 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/

python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 0.55 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/

python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 0.5 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/

python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 0.45 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/

python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 0.4 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/


python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 0.35 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/

python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 0.3 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/

python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --transformer_type bert \
  --alpha 0.25 \
  --use_mca 1 \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_repeat 10 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/

