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
import numpy as np
from datetime import datetime as dt
import os
import pandas as pd
import random


class Train:

    def __init__(self, hyperparameters, config, run_name=None):
        if run_name is None:
            run_name = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = run_name

        # sweep hyperparameters
        self.learning_rate = hyperparameters["learning_rate"] if "learning_rate" in hyperparameters else config.learning_rate
        self.weight_decay = hyperparameters["weight_decay"] if "weight_decay" in hyperparameters else config.weight_decay
        self.batch_size = hyperparameters["batch_size"] if "batch_size" in hyperparameters else config.batch_size
        self.max_epochs = hyperparameters["max_epochs"] if "max_epochs" in hyperparameters else config.max_epochs
        self.lora_use_linear_layers = hyperparameters["lora_use_linear_layers"] if "lora_use_linear_layers" in hyperparameters else config.lora_use_linear_layers

        # config parameters
        self.early_stopping_patience = config.early_stopping_patience
        self.model_name = config.model_name
        self.use_lora = config.use_lora
        self.r = config.r
        self.lora_alpha = config.lora_alpha
        self.lora_dropout = config.lora_dropout
        self.lora_bias = config.lora_bias
        self.use_sampling = config.use_sampling
        self.sample_size = config.sample_size
        self.do_test = config.do_test
        self.num_samples_to_show = config.num_samples_to_show
        self.root_save_dir = config.root_save_dir
        self.save_dir = os.path.join(self.root_save_dir, self.run_name)
        self.log_steps = config.log_steps
        self.save_checkpoint_limit = config.save_checkpoint_limit
        self.data_seed = config.data_seed

        # save configuration
        self.save_configuration(os.path.join(self.save_dir, "configuration.txt"))

    def save_configuration(self, path):
        # create directories that do not exist in the path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(f"run_name: {self.run_name}\n")
            f.write(f"learning_rate: {self.learning_rate}\n")
            f.write(f"weight_decay: {self.weight_decay}\n")
            f.write(f"batch_size: {self.batch_size}\n")
            f.write(f"max_epochs: {self.max_epochs}\n")
            f.write(f"early_stopping_patience: {self.early_stopping_patience}\n")
            f.write(f"model_name: {self.model_name}\n")
            f.write(f"use_lora: {self.use_lora}\n")
            f.write(f"r: {self.r}\n")
            f.write(f"lora_alpha: {self.lora_alpha}\n")
            f.write(f"lora_dropout: {self.lora_dropout}\n")
            f.write(f"lora_bias: {self.lora_bias}\n")
            f.write(f"lora_use_linear_layers: {self.lora_use_linear_layers}\n")
            f.write(f"use_sampling: {self.use_sampling}\n")
            f.write(f"sample_size: {self.sample_size}\n")
            f.write(f"do_test: {self.do_test}\n")
            f.write(f"num_samples_to_show: {self.num_samples_to_show}\n")
            f.write(f"root_save_dir: {self.root_save_dir}\n")
            f.write(f"save_dir: {self.save_dir}\n")
            f.write(f"log_steps: {self.log_steps}\n")
            f.write(f"save_checkpoint_limit: {self.save_checkpoint_limit}\n")
            f.write(f"data_seed: {self.data_seed}\n")


    def run(self):
        self.load_and_prepare_dataset()
        self.set_training_arguments()
        self.set_metrics()
        # create model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=len(self.label2id))
        if self.use_lora:
            if self.lora_use_linear_layers:
                self.lora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS, inference_mode=False, r=self.r, lora_alpha=self.lora_alpha, 
                    lora_dropout=self.lora_dropout, bias=self.lora_bias, target_modules=["query", "key", "value", "dense"]
                )
            else:
                self.lora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS, inference_mode=False, r=self.r, lora_alpha=self.lora_alpha, 
                    lora_dropout=self.lora_dropout, bias=self.lora_bias
                )
            self.model = get_peft_model(self.model, self.lora_config)
            # save trainable parameters count
            with open(os.path.join(self.save_dir, "trainable_params.txt"), "w") as f:
                f.write(self.get_trainable_parameters(self.model))

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
        log_df = pd.DataFrame(trainer.state.log_history)
        # save log history
        log_df.to_csv(os.path.join(self.save_dir, "log.csv"), index=False)
        # get best evaluation accuracy
        best_eval_acc = log_df["eval_accuracy"].max()
        # get step of best evaluation accuracy
        best_eval_acc_step = log_df[log_df["eval_accuracy"] == best_eval_acc]["step"].values[0]
        # get best evaluation loss
        best_eval_loss = log_df["eval_loss"].min()
        # get best evaluation f1
        best_eval_f1 = log_df["eval_f1"].max()
        metrics = {
            "best_eval_acc": best_eval_acc,
            "best_eval_acc_step": best_eval_acc_step,
            "best_eval_loss": best_eval_loss,
            "best_eval_f1": best_eval_f1
        }

        # testing
        if self.do_test:
            test_metrics = self.test(trainer)
            metrics.update(test_metrics)
          
        return metrics
    
    def test(self, trainer):
        predictions, label_ids, test_metrics = trainer.predict(self.dataset["test"])
        # get predicted labels from logits
        predicted_labels = np.argmax(predictions, axis=1)
        correct_predictions_indexes = np.where(predicted_labels == label_ids)[0]
        wrong_predictions_indexes = np.where(predicted_labels != label_ids)[0]
        # sample correct predictions
        num_correct_to_sample = min(self.num_samples_to_show, len(correct_predictions_indexes))
        num_wrong_to_sample = min(self.num_samples_to_show, len(wrong_predictions_indexes))
        correct_predictions_indexes = random.sample(list(correct_predictions_indexes), num_correct_to_sample)
        wrong_predictions_indexes = random.sample(list(wrong_predictions_indexes), num_wrong_to_sample)
        # get correct predictions
        correct_predictions_input = self.dataset["test"][correct_predictions_indexes]
        # decode input_ids of correct predictions to text
        correct_sentences = self.tokenizer.batch_decode(correct_predictions_input["input_ids"], skip_special_tokens=True)
        # create dataframe of correct prediction sentences, predicted labels, and true labels
        correct_predictions_df = pd.DataFrame({
            "sentence": correct_sentences,
            "predicted_label": [self.id2label[label_id] for label_id in predicted_labels[correct_predictions_indexes]],
            "true_label": [self.id2label[label_id] for label_id in label_ids[correct_predictions_indexes]]
        })
        # get wrong predictions
        wrong_predictions_input = self.dataset["test"][wrong_predictions_indexes]
        # decode input_ids of wrong predictions to text
        wrong_sentences = self.tokenizer.batch_decode(wrong_predictions_input["input_ids"], skip_special_tokens=True)
        # create dataframe of wrong prediction sentences, predicted labels, and true labels
        wrong_predictions_df = pd.DataFrame({
            "sentence": wrong_sentences,
            "predicted_label": [self.id2label[label_id] for label_id in predicted_labels[wrong_predictions_indexes]],
            "true_label": [self.id2label[label_id] for label_id in label_ids[wrong_predictions_indexes]]
        })
        # save test results
        correct_predictions_df.to_csv(os.path.join(self.save_dir, "correct_predictions.csv"), index=False)
        wrong_predictions_df.to_csv(os.path.join(self.save_dir, "wrong_predictions.csv"), index=False)

        return test_metrics

    def set_training_arguments(self):
        self.training_args = TrainingArguments(
            output_dir=self.save_dir,
            learning_rate=self.learning_rate,
            num_train_epochs = self.max_epochs,
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
        emotion_dataset = load_dataset("dair-ai/emotion", cache_dir="./cached_datasets")
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
    
    def get_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        