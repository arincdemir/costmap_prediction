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
            'values': [128]
        },
        'latent_dim': {
            'values': [256, 384]
        },
        'dropout_rate': {
            'min': 0.1,
            'max': 0.16
        },
        'cnn_channels': {
            'values': [
                [32, 64, 128],
                [63, 128, 256]
            ]
        },
        'encoder_hidden_dims': {
            'values': [
                [256, 256],
                [512, 512]
            ]
        },
        'decoder_hidden_dims': {
            'values': [
                [256, 512, 512],
                [512, 1024, 1024]
            ]
        },
    }
}

# Parameters that shouldn't change during sweeps (defaults)
default_params = {
    't_dim': 1,
    'grid_size': 32,
    'num_epochs': 4000,
    'dataset_size': 5120,
    'dataset_population_factor': 4,
    'early_stopping_patience': 25,
    'early_stopping_min_delta': 0.000001,
}