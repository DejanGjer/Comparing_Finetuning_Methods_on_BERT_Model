import wandb

sweep_config = {
    'project': 'lora-emotion-classification',
    'method': 'grid',
    'metric': {
      'name': 'eval/accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'learning_rate': {
            'values': [5e-5, 3e-5, 2e-5]
        },
        'weight_decay': {
            'values': [0.01, 0.001]
        },
        'batch_size': {
            'value': 16
        }
    }
}

