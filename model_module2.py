import math
import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class SubjectDiscriminator(nn.Module):
    def __init__(self, feature_dim, num_subjects=30):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_subjects)
        )
    
    def forward(self, x):
        return self.classifier(x)


class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)
        out = self.to_out(out)
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.embed_layer = nn.Sequential(nn.Conv2d(1, d_model * 4, kernel_size=(1, 8), padding='same'),
                                         nn.BatchNorm2d(d_model * 4),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(d_model * 4, d_model, kernel_size=(c_in, 1), padding='valid'),
            nn.BatchNorm2d(d_model),
            nn.GELU())
        self.position_embedding = PositionalEmbedding(d_model)

    def forward(self, x, return_skips=False):
        x = x.unsqueeze(1)  # [batch, 1, 32, 128]
        skip1_raw = self.embed_layer(x)  # [batch, 64, 32, 128] -> skip connection
        x = self.embed_layer2(skip1_raw).squeeze(2)  # [batch, 16, 128] -> skip connection
        skip2 = x.clone()  # [batch, 16, 128]
        x = x.permute(0, 2, 1)  # [batch, 128, 16]
        x = x + self.position_embedding(x)
        
        if return_skips:
            # Format skip connections for U-Net decoder
            # skip1: [batch, 64, 32, 128] -> [batch, 64*32, 128] (flatten spatial)
            batch_size, channels, spatial, temporal = skip1_raw.shape
            skip1_formatted = skip1_raw.view(batch_size, channels * spatial, temporal)  # [batch, 64*32=2048, 128]
            # skip2: [batch, 16, 128] -> already correct format
            skip2_formatted = skip2  # [batch, 16, 128]
            return x, skip1_formatted, skip2_formatted
        return x


class TokenEmbeddingOriginal(nn.Module):
    """Original TokenEmbedding without SSL modifications"""
    def __init__(self, c_in, d_model):
        super(TokenEmbeddingOriginal, self).__init__()
        self.embed_layer = nn.Sequential(nn.Conv2d(1, d_model * 4, kernel_size=(1, 8), padding='same'),
                                         nn.BatchNorm2d(d_model * 4),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(d_model * 4, d_model, kernel_size=(c_in, 1), padding='valid'),
            nn.BatchNorm2d(d_model),
            nn.GELU())
        self.position_embedding = PositionalEmbedding(d_model)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embed_layer(x)
        x = self.embed_layer2(x).squeeze(2)
        x = x.permute(0, 2, 1)
        x = x + self.position_embedding(x)
        return x


class EEGFullDARNetDecoder(nn.Module):
    """Original heavy decoder for full DARNet SSL approach - 400k+ params"""
    def __init__(self, feature_dim=8, c_out=32, seq_len=128):
        super().__init__()
        self.seq_len = seq_len
        self.c_out = c_out
        
        
        self.feature_expand = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 64 * seq_len)  # 64 channels × 128 time steps
        )
        

        self.temporal_conv1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.temporal_conv2 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Final reconstruction layer
        self.final_conv = nn.Conv1d(64, c_out, kernel_size=1)
        
    def forward(self, features):
        """
        features: [batch, feature_dim] - compressed features from full DARNet pipeline
        Returns: [batch, c_out, seq_len] - reconstructed EEG
        """
        batch_size = features.shape[0]
        
        # Expand features to intermediate representation
        x = self.feature_expand(features)  # [batch, 64 * seq_len]
        
        # Reshape to conv format
        x = x.view(batch_size, 64, self.seq_len)  # [batch, 64, 128]
        
        # Temporal processing
        x = self.temporal_conv1(x)  # [batch, 128, 128]
        x = self.temporal_conv2(x)  # [batch, 64, 128]
        
        # Final reconstruction
        reconstructed = self.final_conv(x)  # [batch, 32, 128]
        
        return reconstructed


# class EEGUNetDecoder(nn.Module):
#     """U-Net style decoder with skip connections for EEG reconstruction"""
#     def __init__(self, feature_dim=16, c_out=32, seq_len=128):
#         super().__init__()
#         self.seq_len = seq_len
        
#         # Decoder layers with transpose convolutions
#         self.upconv1 = nn.ConvTranspose1d(feature_dim, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm1d(32)
        
#         # After concatenation with skip2: 32 + 16 = 48 channels
#         self.conv1 = nn.Conv1d(48, 32, kernel_size=3, padding=1)
#         self.bn1_conv = nn.BatchNorm1d(32)
        
#         self.upconv2 = nn.ConvTranspose1d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm1d(64)
        
#         # After concatenation with skip1: 64 + 2048 = 2112 channels (64*32 from spatial flattening)
#         self.conv2 = nn.Conv1d(64 + 2048, 64, kernel_size=3, padding=1)
#         self.bn2_conv = nn.BatchNorm1d(64)
        
#         # Final reconstruction layer
#         self.final_conv = nn.Conv1d(64, c_out, kernel_size=1)
        
#         self.activation = nn.ReLU()
#         self.dropout = nn.Dropout(0.1)
        
#     def forward(self, features, skip1, skip2):
#         """
#         features: [batch, seq_len, feature_dim] - main features from encoder
#         skip1: [batch, 2048, seq_len] - early encoder features (flattened spatial)
#         skip2: [batch, 16, seq_len] - later encoder features
#         """
#         # Convert features to conv format: [batch, feature_dim, seq_len]
#         x = features.permute(0, 2, 1)  # [batch, 16, 128]
        
#         # First upsampling block
#         x = self.upconv1(x)  # [batch, 32, 128]
#         x = self.bn1(x)
#         x = self.activation(x)
        
#         # Concatenate with skip2 connection
#         x = torch.cat([x, skip2], dim=1)  # [batch, 32+16=48, 128]
#         x = self.conv1(x)  # [batch, 32, 128]
#         x = self.bn1_conv(x)
#         x = self.activation(x)
#         x = self.dropout(x)
        
#         # Second upsampling block
#         x = self.upconv2(x)  # [batch, 64, 128]
#         x = self.bn2(x)
#         x = self.activation(x)
        
#         # Concatenate with skip1 connection
#         x = torch.cat([x, skip1], dim=1)  # [batch, 64+2048=2112, 128]
#         x = self.conv2(x)  # [batch, 64, 128]
#         x = self.bn2_conv(x)
#         x = self.activation(x)
#         x = self.dropout(x)
        
#         # Final reconstruction
#         reconstructed = self.final_conv(x)  # [batch, 32, 128]
        
#         return reconstructed


class Refine(nn.Module):
    def __init__(self, c_in):
        super(Refine, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class MyAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.attention_layer = Attention(emb_size, num_heads, dropout=0.1)

    def forward(self, x):
        x = self.attention_layer(x)
        return x


class AttnRefine(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.attention = MyAttention(emb_size, num_heads)
        self.conv_layer = Refine(emb_size)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(emb_size, 4)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x_src = self.attention(x)
        x_src = self.conv_layer(x_src)
        gap = self.gap(x_src.permute(0, 2, 1))
        out = self.out(self.flatten(gap))
        return x_src, out


class DARNet_Original_SSL(nn.Module):
    """Original DARNet with SSL pre-training capability (no domain adversarial)"""
    def __init__(self, args):
        super().__init__()
        # Parameters Initialization
        channel_size = args.eeg_channel
        d_model = 16
        emb_size = d_model
        num_heads = 8

        # Spatiotemporal encoder (same as original DARNet)
        self.token_embedding = TokenEmbedding(c_in=channel_size, d_model=d_model)
        
        # Attention refinement layers (same as original DARNet)
        self.stack1 = AttnRefine(emb_size, num_heads)
        self.stack2 = AttnRefine(emb_size, num_heads)
        
        # SSL U-Net decoder for masked reconstruction
        seq_len = getattr(args, 'win_len', 128)  # Get sequence length from args
        self.ssl_decoder = EEGUNetDecoder(feature_dim=emb_size, c_out=channel_size, seq_len=seq_len)
        
        # AAD classifier (for fine-tuning)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(8, 2)
        
        # Training mode
        self.training_mode = 'ssl'  # 'ssl' or 'aad'

    def set_training_mode(self, mode):
        """Set training mode: 'ssl' for pre-training, 'aad' for fine-tuning"""
        self.training_mode = mode

    def forward_ssl(self, x):
        """Forward pass for SSL pre-training (masked reconstruction)"""
        # Extract features using spatiotemporal encoder with skip connections
        x_src, skip1, skip2 = self.token_embedding(x, return_skips=True)  # [batch, 128, 16], skips
        
        # Reconstruct original EEG from features using U-Net decoder
        reconstructed = self.ssl_decoder(x_src, skip1, skip2)  # [batch, 32, 128]
        
        return reconstructed

    def forward_aad(self, x):
        """Forward pass for AAD classification (fine-tuning)"""
        # Extract spatiotemporal features (same as original DARNet)
        x_src = self.token_embedding(x)

        new_x = []
        x_src1, new_src1 = self.stack1(x_src)
        new_x.append(new_src1)

        x_src2, new_src2 = self.stack2(x_src1)
        new_x.append(new_src2)

        # Extract features before final classification
        out = torch.cat(new_x, -1)
        out = self.flatten(out)
        out = self.out(out)
        
        return out

    def forward(self, x):
        """Forward pass based on training mode"""
        if self.training_mode == 'ssl':
            return self.forward_ssl(x)
        else:
            return self.forward_aad(x)

    def load_ssl_weights(self, ssl_model_path):
        """Load SSL pre-trained weights for fine-tuning"""
        ssl_state_dict = torch.load(ssl_model_path, map_location='cpu')
        
        # Load only the encoder weights (token_embedding)
        encoder_state_dict = {}
        for key, value in ssl_state_dict.items():
            if key.startswith('token_embedding'):
                encoder_state_dict[key] = value
        
        self.load_state_dict(encoder_state_dict, strict=False)
        print(f"Loaded SSL pre-trained weights from {ssl_model_path}")


# Updated DARNet with SSL flag
class DARNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Parameters Initialization
        channel_size = args.eeg_channel
        d_model = 16
        emb_size = d_model
        num_heads = 8

        # Check if SSL is enabled and approach type
        self.ssl_enabled = getattr(args, 'ssl_enabled', False)
        self.ssl_approach = getattr(args, 'ssl_approach', 'full_darnet')  # 'full_darnet' or 'unet'
        
        if self.ssl_enabled:
            if self.ssl_approach == 'full_darnet':
                # Full DARNet SSL approach - use original token embedding
                self.token_embedding = TokenEmbeddingOriginal(c_in=channel_size, d_model=d_model)
                
                # Full DARNet decoder for masked reconstruction
                seq_len = getattr(args, 'win_len', 128)
                self.ssl_decoder = EEGFullDARNetDecoder(feature_dim=8, c_out=channel_size, seq_len=seq_len)
            else:
                # U-Net SSL approach (legacy)
                self.token_embedding = TokenEmbedding(c_in=channel_size, d_model=d_model)
                
                # SSL U-Net decoder for masked reconstruction
                seq_len = getattr(args, 'win_len', 128)
                self.ssl_decoder = EEGUNetDecoder(feature_dim=emb_size, c_out=channel_size, seq_len=seq_len)
            
            # Training mode
            self.training_mode = 'ssl'  # 'ssl' or 'aad'
        else:
            # Original DARNet without SSL modifications
            self.token_embedding = TokenEmbeddingOriginal(c_in=channel_size, d_model=d_model)
        
        
        # Common components
        self.flatten = nn.Flatten()
        self.out = nn.Linear(8, 2)
        self.stack1 = AttnRefine(emb_size, num_heads)
        self.stack2 = AttnRefine(emb_size, num_heads)

    def set_training_mode(self, mode):
        """Set training mode: 'ssl' for pre-training, 'aad' for fine-tuning"""
        if self.ssl_enabled:
            self.training_mode = mode

    def forward_ssl(self, x):
        """Forward pass for SSL pre-training (masked reconstruction)"""
        if not self.ssl_enabled:
            raise ValueError("SSL forward pass called but SSL is not enabled!")
            
        if self.ssl_approach == 'full_darnet':
            # Full DARNet SSL: Use complete pipeline up to feature concatenation
            # Extract spatiotemporal features (same as AAD forward)
            x_src = self.token_embedding(x)

            new_x = []
            x_src1, new_src1 = self.stack1(x_src)
            new_x.append(new_src1)

            x_src2, new_src2 = self.stack2(x_src1)
            new_x.append(new_src2)

            # Concatenate features before classification (compressed representation)
            features = torch.cat(new_x, -1)  # [batch, 8]
            features_flat = self.flatten(features)  # [batch, 8]
            
            # Reconstruct original EEG from compressed features
            reconstructed = self.ssl_decoder(features_flat)  # [batch, 32, 128]
        else:
            # U-Net SSL approach (legacy)
            # Extract features using spatiotemporal encoder with skip connections
            x_src, skip1, skip2 = self.token_embedding(x, return_skips=True)
            
            # Reconstruct original EEG from features using U-Net decoder
            reconstructed = self.ssl_decoder(x_src, skip1, skip2)
        
        return reconstructed

    def forward_aad(self, x):
        """Forward pass for AAD classification"""
        # Extract spatiotemporal features
        x_src = self.token_embedding(x)

        new_x = []
        x_src1, new_src1 = self.stack1(x_src)
        new_x.append(new_src1)

        x_src2, new_src2 = self.stack2(x_src1)
        new_x.append(new_src2)

        # Extract features before final classification
        out = torch.cat(new_x, -1)
        out = self.flatten(out)
        out = self.out(out)
        
        return out

    def forward(self, x):
        """Forward pass based on SSL setting and training mode"""
        if self.ssl_enabled and self.training_mode == 'ssl':
            return self.forward_ssl(x)
        else:
            return self.forward_aad(x)

    def load_ssl_weights(self, ssl_model_path):
        """Load SSL pre-trained weights for fine-tuning"""
        if not self.ssl_enabled:
            print("Warning: Trying to load SSL weights but SSL is not enabled!")
            return
            
        ssl_state_dict = torch.load(ssl_model_path, map_location='cpu')
        
        # Load only the encoder weights (token_embedding)
        encoder_state_dict = {}
        for key, value in ssl_state_dict.items():
            if key.startswith('token_embedding'):
                encoder_state_dict[key] = value
        
        self.load_state_dict(encoder_state_dict, strict=False)
        print(f"Loaded SSL pre-trained weights from {ssl_model_path}")