import torch
import torch.nn as nn
import torch.optim as optim

class CNN_CNMP(nn.Module):
    def __init__(self, t_dim: int, grid_size: int, encoder_hidden_dims: list[int], 
                 decoder_hidden_dims: list[int], latent_dim: int, cnn_channels: list[int],
                 dropout_rate: float = 0.2):
        # t_dim: dimension of time indices
        # grid_size: size of input grid (grid_size x grid_size)
        # encoder_hidden_dims: list of encoder hidden layer dimensions
        # decoder_hidden_dims: list of decoder hidden layer dimensions
        # latent_dim: dimension of the latent representation
        # cnn_channels: list of channel dimensions for CNN layers
        super(CNN_CNMP, self).__init__()
        
        self.t_dim = t_dim
        self.grid_size = grid_size
        
        # CNN with regularization (batch norm + dropout)
        cnn_layers = []
        in_channels = 1  # Input: [batch_size, 1, grid_size, grid_size]
        for out_channels in cnn_channels:
            cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))  # Output: [batch_size, out_channels, grid_size, grid_size]
            cnn_layers.append(nn.BatchNorm2d(out_channels))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout2d(dropout_rate))
            cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Output: [batch_size, out_channels, grid_size/2, grid_size/2]
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate the feature map size after CNN
        reduced_size = grid_size // (2 ** len(cnn_channels))  # Spatial dimension after all pooling layers
        self.reduced_size = reduced_size
        self.SM_dim = reduced_size * reduced_size * cnn_channels[-1]  # Total feature dimension after CNN
        
        # Encoder with regularization
        encoder_layers = []
        input_dim = t_dim + self.SM_dim  # Input: [batch_size, t_dim + SM_dim]
        for hidden_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))  # Output: [batch_size, hidden_dim]
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        encoder_layers.append(nn.Linear(input_dim, latent_dim))  # Output: [batch_size, latent_dim]
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder MLP with regularization
        decoder_mlp = []
        input_dim = latent_dim + t_dim  # Input: [batch_size, latent_dim + t_dim]
        for hidden_dim in decoder_hidden_dims:
            decoder_mlp.append(nn.Linear(input_dim, hidden_dim))  # Output: [batch_size, hidden_dim]
            decoder_mlp.append(nn.BatchNorm1d(hidden_dim))
            decoder_mlp.append(nn.ReLU())
            decoder_mlp.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        initial_channels = cnn_channels[-1]
        decoder_mlp.append(nn.Linear(input_dim, reduced_size * reduced_size * initial_channels))  # Output: [batch_size, reduced_size * reduced_size * initial_channels]
        self.decoder_mlp = nn.Sequential(*decoder_mlp)
        
        # Transposed Convolution layers with regularization
        deconv_layers = []
        reversed_channels = [cnn_channels[i] for i in range(len(cnn_channels)-1, -1, -1)]
        reversed_channels.append(1)  # Final output channel
        
        for i in range(len(reversed_channels)-1):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i+1]
            deconv_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))  # Input: [batch_size, in_ch, h, w], Output: [batch_size, out_ch, h*2, w*2]
            if i < len(reversed_channels)-2:
                deconv_layers.append(nn.BatchNorm2d(out_ch))
                deconv_layers.append(nn.ReLU())
                deconv_layers.append(nn.Dropout2d(dropout_rate * 0.5))
        deconv_layers.append(nn.Sigmoid())  # Final output: [batch_size, 1, grid_size, grid_size]
        self.deconv = nn.Sequential(*deconv_layers)

    def preprocess_grid(self, grid):
        # grid: [batch_size, max_encode_size, grid_size, grid_size]
        batch_size, max_encode_size = grid.shape[0], grid.shape[1]
        grid_reshaped = grid.view(-1, 1, self.grid_size, self.grid_size)  # [batch_size*max_encode_size, 1, grid_size, grid_size]
        grid_features = self.cnn(grid_reshaped)  # [batch_size*max_encode_size, cnn_channels[-1], reduced_size, reduced_size]
        grid_features = grid_features.view(batch_size, max_encode_size, -1)  # [batch_size, max_encode_size, cnn_channels[-1]*reduced_size*reduced_size]
        return grid_features

    def forward(self, padded_time_indices, padded_grids, encodings_mask, padded_query_indices, queries_mask):
        # padded_time_indices: [batch_size, max_encode_size, t_dim]
        # padded_grids: [batch_size, max_encode_size, grid_size, grid_size]
        # encodings_mask: [batch_size, max_encode_size]
        # padded_query_indices: [batch_size, max_query_size, t_dim]
        # queries_mask: [batch_size, max_query_size]
        
        grid_features = self.preprocess_grid(padded_grids)  # [batch_size, max_encode_size, cnn_channels[-1]*reduced_size*reduced_size]
        padded_encodings = torch.cat([padded_time_indices, grid_features], dim=-1)  # [batch_size, max_encode_size, t_dim + cnn_channels[-1]*reduced_size*reduced_size]
        
        # Need to reshape for batch norm in encoder
        batch_size, seq_len, feat_dim = padded_encodings.shape
        padded_encodings_reshaped = padded_encodings.reshape(-1, feat_dim)  # [batch_size*max_encode_size, t_dim + SM_dim]
        latent = self.encoder(padded_encodings_reshaped)  # [batch_size*max_encode_size, latent_dim]
        latent = latent.view(batch_size, seq_len, -1)  # [batch_size, max_encode_size, latent_dim]
        
        latent_sum = torch.sum(latent * encodings_mask.unsqueeze(-1), dim=1)  # [batch_size, latent_dim]
        count = torch.clamp(encodings_mask.sum(dim=1, keepdim=True), min=1.0)  # [batch_size, 1]
        latent_mean = latent_sum / count  # [batch_size, latent_dim]
        latent_mean_expanded = latent_mean.unsqueeze(1).expand(-1, padded_query_indices.size(1), -1)  # [batch_size, max_query_size, latent_dim]
        
        decoder_input = torch.cat((latent_mean_expanded, padded_query_indices), dim=-1)  # [batch_size, max_query_size, latent_dim + t_dim]
        batch_size, max_query_size, feat_dim = decoder_input.shape
        
        # Need to reshape for batch norm in decoder MLP
        decoder_input_reshaped = decoder_input.reshape(-1, feat_dim)  # [batch_size*max_query_size, latent_dim + t_dim]
        feature_maps = self.decoder_mlp(decoder_input_reshaped)  # [batch_size*max_query_size, reduced_size*reduced_size*initial_channels]
        
        initial_channels = feature_maps.size(-1) // (self.reduced_size * self.reduced_size)
        feature_maps = feature_maps.view(batch_size * max_query_size, initial_channels, self.reduced_size, self.reduced_size)  # [batch_size*max_query_size, initial_channels, reduced_size, reduced_size]
        output = self.deconv(feature_maps)  # [batch_size*max_query_size, 1, grid_size, grid_size]
        output = output.view(batch_size, max_query_size, self.grid_size, self.grid_size)  # [batch_size, max_query_size, grid_size, grid_size]
        return output

    def loss(self, output, padded_query_targets, queries_mask):
        # output: [batch_size, max_query_size, grid_size, grid_size]
        # padded_query_targets: [batch_size, max_query_size, grid_size, grid_size]
        # queries_mask: [batch_size, max_query_size]
        mse = torch.nn.functional.mse_loss(output, padded_query_targets, reduction='none')  # [batch_size, max_query_size, grid_size, grid_size]
        mse = mse.mean(dim=(-2, -1))  # [batch_size, max_query_size]
        mse = mse * queries_mask  # [batch_size, max_query_size]
        valid_queries = queries_mask.sum().clamp(min=1e-6)  # scalar
        return mse.sum() / valid_queries  # scalar