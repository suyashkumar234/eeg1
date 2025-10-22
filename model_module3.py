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


class EEGTransformerDecoder(nn.Module):
    """Lightweight Transformer decoder for EEG reconstruction"""
    def __init__(self, feature_dim=8, c_out=32, seq_len=128, d_model=64, num_heads=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.c_out = c_out
        self.d_model = d_model
        
        # Project compressed features to decoder dimension
        self.feature_projection = nn.Linear(feature_dim, d_model)
        
        # Learnable query tokens for reconstruction
        self.query_tokens = nn.Parameter(torch.randn(c_out, d_model))  # [32, 64]
        
        # Positional embedding for queries
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model))  # [128, 64]
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Final projection to time series
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, seq_len),  # [64] -> [128]
            nn.Tanh()  # Bounded output for EEG signals
        )
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, features):
        """
        features: [batch, feature_dim] - compressed features from full DARNet pipeline
        Returns: [batch, c_out, seq_len] - reconstructed EEG
        """
        batch_size = features.shape[0]
        
        # Project features to decoder dimension
        features_proj = self.feature_projection(features)  # [batch, d_model]
        features_proj = features_proj.unsqueeze(1)  # [batch, 1, d_model]
        
        # Prepare query tokens for each batch
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 32, d_model]
        
        # Add positional embeddings to queries (using first seq_len positions)
        pos_emb = self.pos_embedding[:self.c_out].unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 32, d_model]
        queries = queries + pos_emb
        
        # Apply transformer layers
        x = queries
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms):
            # Self-attention + cross-attention with features
            attn_out, _ = attention(x, features_proj, features_proj)
            x = layer_norm(x + self.dropout(attn_out))
            
            # Cross-attention with compressed features
            cross_attn_out, _ = attention(x, features_proj, features_proj)
            x = layer_norm(x + self.dropout(cross_attn_out))
        
        # Project to time series
        reconstructed = self.output_projection(x)  # [batch, 32, 128]
        
        return reconstructed


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


class DARNet_ViT_Decoder(nn.Module):
    """DARNet with ViT/Transformer decoder for SSL pre-training"""
    def __init__(self, args):
        super().__init__()
        # Parameters Initialization
        channel_size = args.eeg_channel
        d_model = 16
        emb_size = d_model
        num_heads = 8

        # Spatiotemporal encoder (same as original DARNet)
        self.token_embedding = TokenEmbeddingOriginal(c_in=channel_size, d_model=d_model)
        
        # Attention refinement layers (same as original DARNet)
        self.stack1 = AttnRefine(emb_size, num_heads)
        self.stack2 = AttnRefine(emb_size, num_heads)
        
        # Lightweight Transformer decoder for masked reconstruction
        seq_len = getattr(args, 'win_len', 128)
        self.ssl_decoder = EEGTransformerDecoder(
            feature_dim=8, 
            c_out=channel_size, 
            seq_len=seq_len,
            d_model=64,  # Transformer dimension
            num_heads=4,  # Fewer heads for efficiency
            num_layers=2  # Fewer layers for efficiency
        )
        
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
        # Use complete DARNet pipeline up to feature concatenation
        x_src = self.token_embedding(x)

        new_x = []
        x_src1, new_src1 = self.stack1(x_src)
        new_x.append(new_src1)

        x_src2, new_src2 = self.stack2(x_src1)
        new_x.append(new_src2)

        # Concatenate features before classification (compressed representation)
        features = torch.cat(new_x, -1)  # [batch, 8]
        features_flat = self.flatten(features)  # [batch, 8]
        
        # Reconstruct original EEG from compressed features using Transformer decoder
        reconstructed = self.ssl_decoder(features_flat)  # [batch, 32, 128]
        
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
        
        # Load only the encoder weights (token_embedding, stack1, stack2)
        encoder_state_dict = {}
        for key, value in ssl_state_dict.items():
            if key.startswith(('token_embedding', 'stack1', 'stack2')):
                encoder_state_dict[key] = value
        
        self.load_state_dict(encoder_state_dict, strict=False)
        print(f"Loaded SSL pre-trained weights from {ssl_model_path}")