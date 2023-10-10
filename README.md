# Emotion Classification From Sentences With LLMs using LoRA

In this project for university I used BERT mdoel in order to classify sentences from Twitter based on emotions (joy, sadness, anger, fear, surprise and love). The main goal of this work was to compare performances between two types of fine-tunning: 
- Full fine-tunning - which fine-tunes all the parameters of the model
- LoRA - which uses low rank adaptation to train less parameters

All results of this work you can find in the work for university (in serbian) in the file `rad/ml_rad.pdf`.

## Files

- `code/` - contains all the code for this project
    - `config.py` - run configuration
    - `main.py` - main file for running the project
    - `train.py` - file for training the model
    - `wandb_config.py` - configuration of sweep parameters for wandb
    - `run.sh` - bash script for running the project

- `rad/` - contains the work for university (in serbian)
    - `ml_rad.pdf` - work for university (in serbian)