sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'validation_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.0001,
            'max': 0.001
        },
        'batch_size': {
            'values': [32]
        },
        'latent_dim': {
            'values': [256]
        },
        'dropout_rate': {
            'min': 0.1,
            'max': 0.11
        },
        'cnn_channels': {
            'values': [
                [256, 128, 64],
            ]
        },
        'encoder_hidden_dims': {
            'values': [
                [256,256,256]
            ]
        },
        'decoder_hidden_dims': {
            'values': [
                [256,256,256,256]
            ]
        },
    }
}

# Parameters that shouldn't change during sweeps (defaults)
default_params = {
    't_dim': 1,
    'grid_size': 32,
    'num_epochs': 20000,
    'dataset_size': 10240,
    'early_stopping_patience': 25,
    'early_stopping_min_delta': 0.000001,
}