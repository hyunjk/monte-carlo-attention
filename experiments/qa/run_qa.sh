export SQUAD_DIR=~/Workspace/Datasets/squad_v2

python run_qa.py \
    --model_name_or_path deepset/bert-base-cased-squad2 \
    --dataset_name squad_v2 \
    --do_eval \
    --version_2_with_negative \
    --learning_rate 3e-5 \
    --num_train_epochs 4 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./approx2/ \
    --per_device_eval_batch_size=1  \
    --per_device_train_batch_size=1   \
    --save_steps 5000