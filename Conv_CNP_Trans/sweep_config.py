sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'validation_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.001,
            'max': 0.002
        },
        'batch_size': {
            'values': [32]
        },
        'latent_dim': {
            'values': [128, 190]
        },
        'dropout_rate': {
            'min': 0.15,
            'max': 0.22
        },
        'cnn_channels': {
            'values': [
                [16, 32, 64],
                [16, 32, 32]
            ]
        },
        'encoder_hidden_dims': {
            'values': [
                [128, 128],
                [128, 256]
            ]
        },
        'decoder_hidden_dims': {
            'values': [
                [128, 256, 256],
                [128, 128, 128]
            ]
        },
    }
}

# Parameters that shouldn't change during sweeps (defaults)
default_params = {
    't_dim': 1,
    'grid_size': 32,
    'num_epochs': 1000,
    'dataset_size': 5120,
    'early_stopping_patience': 25,
    'early_stopping_min_delta': 0.000001,
}