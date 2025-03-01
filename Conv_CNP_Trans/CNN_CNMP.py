import torch
import torch.nn as nn
import torch.optim as optim

class CNN_CNMP(nn.Module):
    def __init__(self, t_dim: int, grid_size: int, encoder_hidden_dims: list[int], 
                 decoder_hidden_dims: list[int], latent_dim: int, cnn_channels: list[int]):
        super(CNN_CNMP, self).__init__()
        
        self.t_dim = t_dim
        self.grid_size = grid_size
        
        # CNN to process the grid
        cnn_layers = []
        in_channels = 1 
        for out_channels in cnn_channels:
            cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate the feature map size after CNN
        reduced_size = grid_size // (2 ** len(cnn_channels))
        self.reduced_size = reduced_size
        self.SM_dim = reduced_size * reduced_size * cnn_channels[-1]
        
        # Encoder
        encoder_layers = []
        input_dim = t_dim + self.SM_dim
        for hidden_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            input_dim = hidden_dim
        encoder_layers.append(nn.Linear(input_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder MLP to convert latent representation to feature map
        decoder_mlp = []
        input_dim = latent_dim + t_dim
        for hidden_dim in decoder_hidden_dims:
            decoder_mlp.append(nn.Linear(input_dim, hidden_dim))
            decoder_mlp.append(nn.ReLU())
            input_dim = hidden_dim
        initial_channels = cnn_channels[-1]  # Start with same number of channels as CNN output
        decoder_mlp.append(nn.Linear(input_dim, reduced_size * reduced_size * initial_channels))
        self.decoder_mlp = nn.Sequential(*decoder_mlp)
        
        # Transposed Convolution layers - mirror the CNN structure in reverse
        deconv_layers = []
        reversed_channels = [cnn_channels[i] for i in range(len(cnn_channels)-1, -1, -1)]
        reversed_channels.append(1)  # Final output channel
        
        for i in range(len(reversed_channels)-1):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i+1]
            deconv_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            if i < len(reversed_channels)-2:  # add activation except after final layer
                deconv_layers.append(nn.ReLU())
        deconv_layers.append(nn.Sigmoid())
        self.deconv = nn.Sequential(*deconv_layers)

    def preprocess_grid(self, grid):
        batch_size, max_encode_size = grid.shape[0], grid.shape[1]
        grid_reshaped = grid.view(-1, 1, self.grid_size, self.grid_size)
        grid_features = self.cnn(grid_reshaped)
        grid_features = grid_features.view(batch_size, max_encode_size, -1)
        return grid_features

    def forward(self, padded_time_indices, padded_grids, encodings_mask, padded_query_indices, queries_mask):
        grid_features = self.preprocess_grid(padded_grids)
        padded_encodings = torch.cat([padded_time_indices, grid_features], dim=-1)
        latent = self.encoder(padded_encodings)
        latent_sum = torch.sum(latent * encodings_mask.unsqueeze(-1), dim=1)
        count = torch.clamp(encodings_mask.sum(dim=1, keepdim=True), min=1.0)
        latent_mean = latent_sum / count
        latent_mean_expanded = latent_mean.unsqueeze(1).expand(-1, padded_query_indices.size(1), -1)
        decoder_input = torch.cat((latent_mean_expanded, padded_query_indices), dim=-1)
        batch_size, max_query_size = decoder_input.shape[0], decoder_input.shape[1]
        feature_maps = self.decoder_mlp(decoder_input)
        initial_channels = feature_maps.size(-1) // (self.reduced_size * self.reduced_size)
        feature_maps = feature_maps.view(batch_size * max_query_size, initial_channels, self.reduced_size, self.reduced_size)
        output = self.deconv(feature_maps)
        output = output.view(batch_size, max_query_size, self.grid_size, self.grid_size)
        return output

    def loss(self, output, padded_query_targets, queries_mask):
        mse = torch.nn.functional.mse_loss(output, padded_query_targets, reduction='none')
        mse = mse.mean(dim=(-2, -1))
        mse = mse * queries_mask
        valid_queries = queries_mask.sum().clamp(min=1e-6)
        return mse.sum() / valid_queries