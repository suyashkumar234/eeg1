import math
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F


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


class ContrastiveHead(nn.Module):
    """Contrastive learning head for attention direction clustering"""
    def __init__(self, feature_dim=8, projection_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, projection_dim)
        )
    
    def forward(self, x):
        return self.projection(x)


class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # print("seq_len",seq_len,self.num_heads)
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale

        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
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

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.embed_layer(x)
        x = self.embed_layer2(x).squeeze(2)
        x = x.permute(0, 2, 1)
        x = x + self.position_embedding(x)
        return x


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
        # print(out.shape)
        return x_src, out


def contrastive_loss(features, attention_labels, temperature=0.1):
    """
    InfoNCE loss for attention direction clustering
    features: [batch, feature_dim] - features for contrastive learning
    attention_labels: [batch] - 0 for left, 1 for right attention
    temperature: temperature parameter for softmax
    """
    batch_size = features.shape[0]
    device = features.device
    
    # Normalize features
    features = F.normalize(features, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(features, features.T) / temperature
    
    # Create positive mask (same attention direction)
    labels_expanded = attention_labels.unsqueeze(1)
    positive_mask = (labels_expanded == labels_expanded.T).float()
    positive_mask.fill_diagonal_(0)  # Remove self-similarity
    
    # Create negative mask (different attention direction + remove diagonal)
    negative_mask = 1 - positive_mask - torch.eye(batch_size, device=device)
    
    # InfoNCE loss computation
    exp_sim = torch.exp(sim_matrix)
    exp_sim = exp_sim * (1 - torch.eye(batch_size, device=device))  # Remove self-similarity
    
    # For each sample, compute positive and negative similarities
    pos_sim = (exp_sim * positive_mask).sum(dim=1)
    neg_sim = (exp_sim * negative_mask).sum(dim=1)
    
    # InfoNCE loss: -log(pos_sim / (pos_sim + neg_sim))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    loss = -torch.log((pos_sim + epsilon) / (pos_sim + neg_sim + epsilon))
    
    # Only compute loss for samples that have positive pairs
    valid_mask = positive_mask.sum(dim=1) > 0
    if valid_mask.sum() > 0:
        loss = loss[valid_mask].mean()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss


class DARNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size = args.eeg_channel
        d_model = 16
        emb_size = d_model
        num_heads = 8

        self.token_embedding = TokenEmbedding(c_in=channel_size, d_model=d_model)

        self.flatten = nn.Flatten()
        self.out = nn.Linear(8, 2)
        self.stack1 = AttnRefine(emb_size, num_heads)
        self.stack2 = AttnRefine(emb_size, num_heads)
        
        # Domain Adversarial Components
        self.use_domain_adversarial = getattr(args, 'use_domain_adversarial', False)
        if self.use_domain_adversarial:
            self.lambda_domain = getattr(args, 'lambda_domain', 0.1)
            self.gradient_reversal = GradientReversalLayer(lambda_=self.lambda_domain)
            # Get number of subjects from args (MM-AAD: 30, KUL: 16, DTU: 18, AVED: 10)
            num_subjects = getattr(args, 'subject_number', 30)
            if hasattr(args, 'dataset') and args.dataset == 'KUL':
                num_subjects = 16
            elif hasattr(args, 'dataset') and args.dataset == 'MM-AAD':
                num_subjects = 30
            elif hasattr(args, 'dataset') and args.dataset == 'DTU':
                num_subjects = 18
            elif hasattr(args, 'dataset') and (args.dataset == 'AVED-audio' or args.dataset == 'AVED-audiovisual'):
                num_subjects = 10
            self.subject_discriminator = SubjectDiscriminator(feature_dim=8, num_subjects=num_subjects)
        
        # Contrastive Learning Components
        self.use_contrastive = getattr(args, 'use_contrastive', False)
        if self.use_contrastive:
            self.contrastive_head = ContrastiveHead(feature_dim=8, projection_dim=128)

    def forward(self, x):
        x_src = self.token_embedding(x)

        new_x = []
        x_src1, new_src1 = self.stack1(x_src)
        new_x.append(new_src1)

        x_src2, new_src2 = self.stack2(x_src1)
        new_x.append(new_src2)

        # Extract features before final classification
        features = torch.cat(new_x, -1)  # [batch, 8]
        features_flat = self.flatten(features)  # [batch, 8]
        
        # Attention classification (main task)
        attention_logits = self.out(features_flat)
        
        # Case 1: Original DARNet (no domain adversarial, no contrastive)
        if not self.use_domain_adversarial and not self.use_contrastive:
            return attention_logits
        
        # Case 2: Domain adversarial only (no contrastive)
        elif self.use_domain_adversarial and not self.use_contrastive:
            if self.training:
                reversed_features = self.gradient_reversal(features_flat)
                subject_logits = self.subject_discriminator(reversed_features)
                return attention_logits, subject_logits
            else:
                return attention_logits
        
        # Case 3: Contrastive only (no domain adversarial)
        elif not self.use_domain_adversarial and self.use_contrastive:
            if self.training:
                contrastive_features = self.contrastive_head(features_flat)
                return attention_logits, contrastive_features
            else:
                return attention_logits
        
        # Case 4: Both domain adversarial and contrastive
        else:  # self.use_domain_adversarial and self.use_contrastive
            if self.training:
                reversed_features = self.gradient_reversal(features_flat)
                subject_logits = self.subject_discriminator(reversed_features)
                contrastive_features = self.contrastive_head(features_flat)
                return attention_logits, subject_logits, contrastive_features
            else:
                return attention_logits
    
    def update_lambda(self, lambda_value):
        if self.use_domain_adversarial:
            self.gradient_reversal.lambda_ = lambda_value
# import math
# import torch
# from torch import nn
# from torch.autograd import Function


# class GradientReversalFunction(Function):
#     @staticmethod
#     def forward(ctx, x, lambda_):
#         ctx.lambda_ = lambda_
#         return x.view_as(x)
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         output = grad_output.neg() * ctx.lambda_
#         return output, None


# class GradientReversalLayer(nn.Module):
#     def __init__(self, lambda_=1.0):
#         super().__init__()
#         self.lambda_ = lambda_
    
#     def forward(self, x):
#         return GradientReversalFunction.apply(x, self.lambda_)


# class SubjectDiscriminator(nn.Module):
#     def __init__(self, feature_dim, num_subjects=30):
#         super().__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(feature_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, num_subjects)
#         )
    
#     def forward(self, x):
#         return self.classifier(x)


# class Attention(nn.Module):
#     def __init__(self, emb_size, num_heads, dropout):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = emb_size ** -0.5
#         # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
#         self.key = nn.Linear(emb_size, emb_size, bias=False)
#         self.value = nn.Linear(emb_size, emb_size, bias=False)
#         self.query = nn.Linear(emb_size, emb_size, bias=False)

#         self.dropout = nn.Dropout(dropout)
#         self.to_out = nn.LayerNorm(emb_size)

#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape
#         # print("seq_len",seq_len,self.num_heads)
#         k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
#         v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
#         q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
#         # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

#         attn = torch.matmul(q, k) * self.scale

#         attn = nn.functional.softmax(attn, dim=-1)
#         out = torch.matmul(attn, v)
#         # out.shape = (batch_size, num_heads, seq_len, d_head)
#         out = out.transpose(1, 2)
#         # out.shape == (batch_size, seq_len, num_heads, d_head)
#         out = out.reshape(batch_size, seq_len, -1)
#         # out.shape == (batch_size, seq_len, d_model)
#         out = self.to_out(out)
#         return out


# class PositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEmbedding, self).__init__()
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).float()
#         pe.require_grad = False

#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return self.pe[:, :x.size(1)]


# class TokenEmbedding(nn.Module):
#     def __init__(self, c_in, d_model):
#         super(TokenEmbedding, self).__init__()
#         self.embed_layer = nn.Sequential(nn.Conv2d(1, d_model * 4, kernel_size=(1, 8), padding='same'),
#                                          nn.BatchNorm2d(d_model * 4),
#                                          nn.GELU())

#         self.embed_layer2 = nn.Sequential(
#             nn.Conv2d(d_model * 4, d_model, kernel_size=(c_in, 1), padding='valid'),
#             nn.BatchNorm2d(d_model),
#             nn.GELU())
#         self.position_embedding = PositionalEmbedding(d_model)

#     def forward(self, x):

#         x = x.unsqueeze(1)
#         x = self.embed_layer(x)
#         x = self.embed_layer2(x).squeeze(2)
#         x = x.permute(0, 2, 1)
#         x = x + self.position_embedding(x)
#         return x


# class Refine(nn.Module):
#     def __init__(self, c_in):
#         super(Refine, self).__init__()
#         padding = 1 if torch.__version__ >= '1.5.0' else 2
#         self.downConv = nn.Conv1d(in_channels=c_in,
#                                   out_channels=c_in,
#                                   kernel_size=3,
#                                   padding=padding)
#         self.norm = nn.BatchNorm1d(c_in)
#         self.activation = nn.ELU()
#         self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#     def forward(self, x):
#         x = self.downConv(x.permute(0, 2, 1))
#         x = self.norm(x)
#         x = self.activation(x)
#         x = self.maxPool(x)
#         x = x.transpose(1, 2)
#         return x


# class MyAttention(nn.Module):
#     def __init__(self, emb_size, num_heads):
#         super().__init__()
#         self.attention_layer = Attention(emb_size, num_heads, dropout=0.1)

#     def forward(self, x):
#         x = self.attention_layer(x)
#         return x


# class AttnRefine(nn.Module):
#     def __init__(self, emb_size, num_heads):
#         super().__init__()
#         self.attention = MyAttention(emb_size, num_heads)
#         self.conv_layer = Refine(emb_size)
#         self.gap = nn.AdaptiveAvgPool1d(1)
#         self.out = nn.Linear(emb_size, 4)
#         self.flatten = nn.Flatten()

#     def forward(self, x):
#         x_src = self.attention(x)
#         x_src = self.conv_layer(x_src)
#         gap = self.gap(x_src.permute(0, 2, 1))
#         out = self.out(self.flatten(gap))
#         # print(out.shape)
#         return x_src, out


# class DARNet(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         # Parameters Initialization -----------------------------------------------
#         channel_size = args.eeg_channel
#         d_model = 16
#         emb_size = d_model
#         num_heads = 8

#         self.token_embedding = TokenEmbedding(c_in=channel_size, d_model=d_model)

#         self.flatten = nn.Flatten()
#         self.out = nn.Linear(8, 2)
#         self.stack1 = AttnRefine(emb_size, num_heads)
#         self.stack2 = AttnRefine(emb_size, num_heads)
        
#         # Domain Adversarial Components
#         self.use_domain_adversarial = getattr(args, 'use_domain_adversarial', False)
#         if self.use_domain_adversarial:
#             self.lambda_domain = getattr(args, 'lambda_domain', 0.1)
#             self.gradient_reversal = GradientReversalLayer(lambda_=self.lambda_domain)
#             self.subject_discriminator = SubjectDiscriminator(feature_dim=8, num_subjects=30)

#     def forward(self, x):
#         x_src = self.token_embedding(x)

#         new_x = []
#         x_src1, new_src1 = self.stack1(x_src)
#         new_x.append(new_src1)

#         x_src2, new_src2 = self.stack2(x_src1)
#         new_x.append(new_src2)

#         # Extract features before final classification
#         features = torch.cat(new_x, -1)
#         features_flat = self.flatten(features)
        
#         # Attention classification (main task)
#         attention_logits = self.out(features_flat)
        
#         # Subject discrimination (domain adversarial task)
#         if self.use_domain_adversarial and self.training:
#             reversed_features = self.gradient_reversal(features_flat)
#             subject_logits = self.subject_discriminator(reversed_features)
#             return attention_logits, subject_logits
        
#         return attention_logits
    
#     def update_lambda(self, lambda_value):
#         if self.use_domain_adversarial:
#             self.gradient_reversal.lambda_ = lambda_value