import subprocess

pretrained_model = {
    "bert": {
        "cola": "textattack/bert-base-uncased-CoLA",
        "mnli": "ishan/bert-base-uncased-mnli",
        "mrpc": "bert-base-cased-finetuned-mrpc",
        "qnli": "textattack/bert-base-uncased-QNLI",
        "qqp": "textattack/bert-base-uncased-QQP",
        "rte": "textattack/bert-base-uncased-RTE",
        "sst2": "textattack/bert-base-uncased-SST-2",
        "stsb": "textattack/bert-base-uncased-STS-B",
        "wnli": "textattack/bert-base-uncased-WNLI",
    },
    "distilbert": {
        "cola": "textattack/distilbert-base-uncased-CoLA",
        "mnli": "textattack/distilbert-base-uncased-MNLI",
        "mrpc": "textattack/distilbert-base-uncased-MRPC",
        "qnli": "textattack/distilbert-base-uncased-QNLI",
        "qqp": "textattack/distilbert-base-uncased-QQP",
        "rte": "textattack/distilbert-base-uncased-RTE",
        "sst2": "avneet/distilbert-base-uncased-finetuned-sst2",
        "stsb": "eduardofv/stsb-m-mt-es-distilbert-base-uncased",  # 문제 있
        "wnli": "textattack/distilbert-base-uncased-WNLI",
    }
}

alpha_tests = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
repeat = 1

for transformer_type in pretrained_model:

    # if transformer_type != "distilbert":
    #     continue

    for task_name in pretrained_model[transformer_type]:

        if task_name != "sst2":
            continue

        for alpha in alpha_tests:
            if alpha == 0:
                use_mca = False
            else:
                use_mca = True

            model_name = pretrained_model[transformer_type][task_name]

            subprocess.run(["python",
                            "run_glue.py",
                            "--model_name_or_path",
                            model_name,
                            "--transformer_type",
                            transformer_type,
                            "--alpha", f"{alpha}",
                            "--use_mca", f"{use_mca}",
                            "--task_name", task_name,
                            "--do_eval",
                            "--eval_repeat", f"{repeat}",
                            "--max_seq_length", "128",
                            "--per_device_train_batch_size", "1",
                            "--per_device_eval_batch_size", "1",
                            "--learning_rate", "2e-5",
                            "--num_train_epochs", "3",
                            "--output_dir", "./approx/"
                            ], shell=False)
