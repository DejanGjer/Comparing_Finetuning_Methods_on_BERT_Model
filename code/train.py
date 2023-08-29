from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, LoraConfig
import evaluate
import torch
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from datetime import datetime as dt
import os


class Train:

    def __init__(self, hyperparameters, config, run_name=None):
        if run_name is None:
            run_name = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = run_name

        # sweep hyperparameters
        self.learning_rate = hyperparameters["learning_rate"]
        self.weight_decay = hyperparameters["weight_decay"]
        self.batch_size = hyperparameters["batch_size"]

        # config parameters
        self.early_stopping_patience = config.early_stopping_patience
        self.model_name = config.model_name
        self.use_lora = config.use_lora
        self.use_sampling = config.use_sampling
        self.sample_size = config.sample_size
        self.root_save_dir = config.root_save_dir
        self.save_dir = os.path.join(self.root_save_dir, self.run_name)
        self.log_steps = config.log_steps
        self.save_checkpoint_limit = config.save_checkpoint_limit
        self.data_seed = config.data_seed

    def run(self):
        self.load_and_prepare_dataset()
        self.set_training_arguments()
        self.set_metrics()
        # create model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=len(self.label2id))
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=self.early_stopping_patience)]
        )
        trainer.train() 

    def set_training_arguments(self):
        self.training_args = TrainingArguments(
            output_dir=self.save_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            logging_strategy = "steps",
            logging_steps = self.log_steps,
            evaluation_strategy = "steps",
            eval_steps = self.log_steps,
            save_strategy = "steps",
            save_steps = self.log_steps,
            load_best_model_at_end = True,
            metric_for_best_model = "accuracy",
            greater_is_better=True,
            save_total_limit = self.save_checkpoint_limit,
            data_seed = self.data_seed,
            report_to="wandb"  # enable logging to W&B
        )

    # method for tokenizing one batch at a time
    def tokenize(self, batch):
        return self.tokenizer(batch["text"], padding="max_length", truncation=True)

    def load_and_prepare_dataset(self):
        emotion_dataset = load_dataset("dair-ai/emotion")
        self.label2id = {text: num for num, text in enumerate(emotion_dataset["train"].features["label"].names)}
        self.id2label = {num: text for num, text in enumerate(emotion_dataset["train"].features["label"].names)}
        if self.use_sampling:
            emotion_dataset["train"] = emotion_dataset["train"].select(range(self.sample_size))
            emotion_dataset["validation"] = emotion_dataset["validation"].select(range(self.sample_size//10))
            emotion_dataset["test"] = emotion_dataset["test"].select(range(self.sample_size//10))
        print("Dataset is loaded")
        print(emotion_dataset)

        # load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        # tokenize dataset
        tokenized_emotions = emotion_dataset.map(self.tokenize, batched=True, batch_size=self.batch_size)
        tokenized_emotions = tokenized_emotions.remove_columns(["text"])
        tokenized_emotions = tokenized_emotions.rename_column("label", "labels")
        self.dataset = tokenized_emotions

        # define data collator
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding="max_length", return_tensors="pt")

    def set_metrics(self):
        # define metrics
        self.metric_acc = evaluate.load("accuracy")
        self.metric_precision = evaluate.load("precision")
        self.metric_recall = evaluate.load("recall")
        self.metric_f1 = evaluate.load("f1")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": self.metric_acc.compute(predictions=predictions, references=labels)["accuracy"],
            "precision": self.metric_precision.compute(predictions=predictions, references=labels, average="weighted")["precision"],
            "recall": self.metric_recall.compute(predictions=predictions, references=labels, average="weighted")["recall"],
            "f1": self.metric_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
        }