export TASK_NAME=sst2

python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --task_name $TASK_NAME \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./approx/