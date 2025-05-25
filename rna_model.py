import math
import torch

import torch.nn               as nn
import torch.nn.functional    as F
import matplotlib.pyplot      as plt
import torch.utils.checkpoint as checkpoint


from torch                    import einsum
from einops                   import rearrange, repeat, reduce
from einops.layers.torch      import Rearrange
from dropout                  import *


def init_weights(m):
    if m is not None and isinstance(m, nn.Linear):
        pass


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class OuterProductMean(nn.Module):
    def __init__(self, in_dim=256, dim_msa=32, pairwise_dim=64):
        super(OuterProductMean, self).__init__()
        self.proj_down1 = nn.Linear(in_dim, dim_msa)            
        self.proj_down2 = nn.Linear(dim_msa**2, pairwise_dim)

    def forward(self, seq_rep):
        seq_rep       = self.proj_down1(seq_rep)                                # Reducing dimension from 256->32
        outer_product = torch.einsum('bid,bjc -> bijcd', seq_rep, seq_rep)      # Einstein summation : calculate pairwise interactions between sequence positions
        outer_product = rearrange(outer_product, 'b i j c d -> b i j (c d)')    # Rearranges the outer product tensor into shape (batch, seq_len, seq_len, dim_msa^2) using einops
        outer_product = self.proj_down2(outer_product)                          # Maps flattened dimension (32^2=1024) to pairwise_dim=64 via proj_down2

        return outer_product

class OuterProductMeanSimple(nn.Module):

    def __init__(self, in_dim=256, dim_msa=32, pairwise_dim=64):
        super(OuterProductMean, self).__init__()
        self.proj_down1 = nn.Linear(in_dim, dim_msa)            
        self.proj_down2 = nn.Linear(dim_msa**2, pairwise_dim)

    def forward(self, seq_rep):
        seq_rep = self.proj_down1(seq_rep)  # (batch, seq_len, dim_msa)
        batch_size, seq_len, dim = seq_rep.shape
        
        # Initialize output tensor
        outer_product = torch.zeros(batch_size, seq_len, seq_len, dim, dim)
        
        # Iterate over batches and sequence positions
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    vec_i = seq_rep[b, i]  # (dim_msa,)
                    vec_j = seq_rep[b, j]  # (dim_msa,)
                    outer_product[b, i, j] = torch.outer(vec_i, vec_j)  # (dim_msa, dim_msa)
        
        outer_product = rearrange(outer_product, 'b i j c d -> b i j (c d)')
        return self.proj_down2(outer_product)

class RelPos(nn.Module):

    def __init__(self, dim=64):
        super(RelPos, self).__init__()
        self.linear = nn.Linear(17, dim)

    def forward(self, src):
        device      = src.device
        L           = src.shape[1]
        res_id      = torch.arange(L).to(device).unsqueeze(0)           # tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], device='cuda:0')
        bin_values  = torch.arange(-8, 9, device=device)                # tensor([-8, -7, -6, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8], device='cuda:0')
        d           = res_id[:, :, None] - res_id[:, None, :]           # Symmetric metric
        bdy         = torch.tensor(8, device=device)                    # tensor(8, device='cuda:0')
        d           = torch.minimum(torch.maximum(-bdy, d), bdy)        # filtering elements to be the max 8 and min -8
        d_onehot    = (d[..., None] == bin_values).float()              # Binarizing the array 1 mean there is relation between those neucliotides and 0 means not. element wise comparision.

        assert d_onehot.sum(dim=-1).min() == 1
        p           = self.linear(d_onehot)                             # Sends binary matrix to the linar layer which learns relative relationships between neucliotide.
        return p

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, attn_mask=None):
        attn     = torch.matmul(q, k.transpose(2, 3))/ self.temperature
        if mask is not None:
            attn = attn+mask

        if attn_mask is not None:
            for i in range(len(attn_mask)):
                attn_mask[i,0] = attn_mask[i,0].fill_diagonal_(1)
            attn = attn.float().masked_fill(attn_mask == 0, float('-1e-9'))

        attn   = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1): #d_model = 256, n_head=8, d_k=32, d_v=32
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False) # all heads together weight matrix
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None,src_mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        qw = self.w_qs(q)
        kw = self.w_ks(k)
        vw = self.w_vs(v)

        q = qw.view(sz_b, len_q, n_head, d_k) 
        k = kw.view(sz_b, len_k, n_head, d_k)
        v = vw.view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask  # For head axis broadcasting

        if src_mask is not None:
            src_mask=src_mask[:,:q.shape[2]].unsqueeze(-1).float()
            attn_mask=torch.matmul(src_mask,src_mask.permute(0,2,1))#.long()
            attn_mask=attn_mask.unsqueeze(1)
            q, attn = self.attention(q, k, v, mask=mask,attn_mask=attn_mask)
        else:
            q, attn = self.attention(q, k, v, mask=mask)
  
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class TriangleMultiplicativeModule(nn.Module):
    
    def __init__(
        self,
        *,
        dim,
        hidden_dim = None,
        mix = 'ingoing'
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, src_mask):
        src_mask = src_mask.unsqueeze(-1).float()                               # Add dimension to the last 
        mask     = torch.matmul(src_mask,src_mask.permute(0,2,1))               # Matrix multiplication to get the mask for the feature map
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()') # unsqueeze(-1) operation

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)

class ConvTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, pairwise_dimension, dropout=0.1, k=3):
        super(ConvTransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)

        self.linear1        = nn.Linear(d_model, dim_feedforward)
        self.dropout        = nn.Dropout(dropout)
        self.linear2        = nn.Linear(dim_feedforward, d_model)

        self.norm1          = nn.LayerNorm(d_model)
        self.norm2          = nn.LayerNorm(d_model)
        self.norm3          = nn.LayerNorm(d_model)

        self.dropout1       = nn.Dropout(dropout)
        self.dropout2       = nn.Dropout(dropout)
        self.dropout3       = nn.Dropout(dropout)

        self.pairwise2heads = nn.Linear(pairwise_dimension, nhead, bias=False)
        self.pairwise_norm  = nn.LayerNorm(pairwise_dimension)
        self.activation     = nn.GELU()

        self.conv=nn.Conv1d(d_model,d_model,k,padding=k//2)

        self.triangle_update_out = TriangleMultiplicativeModule(dim=pairwise_dimension, mix='outgoing')
        self.triangle_update_in  = TriangleMultiplicativeModule(dim=pairwise_dimension, mix='ingoing')

        self.pair_dropout_out    = DropoutRowwise(dropout)
        self.pair_dropout_in     = DropoutRowwise(dropout)

        self.outer_product_mean = OuterProductMean(in_dim=d_model, pairwise_dim=pairwise_dimension)

        self.pair_transition    = nn.Sequential(
            nn.LayerNorm(pairwise_dimension),
            nn.Linear(pairwise_dimension, pairwise_dimension * 4),
            nn.ReLU(inplace=True),
            nn.Linear(pairwise_dimension * 4, pairwise_dimension)
        )

    def forward(self, src , pairwise_features, src_mask=None):
        src                     = src * src_mask.float().unsqueeze(-1)
        conv_ops                = self.conv(src.permute(0,2,1)).permute(0,2,1)
        src                     = src * conv_ops
        src                     = self.norm3(src) # shape(1,6,256), Normalization on last dimension -> 256

        pairwise_norm           = self.pairwise_norm(pairwise_features) # shape(1,6,6,64), Normalization on last dimension -> 64
        pairwise_bias           = self.pairwise2heads(pairwise_norm).permute(0,3,1,2) # 64 -> 8 mapping by linear layer in pairwise2heads function and then just changing the dimensions.
        src2, attention_weights = self.self_attn(src, src, src, mask=pairwise_bias, src_mask=src_mask)

        src                     = src + self.dropout1(src2)
        src                     = self.norm1(src)
        linear1                 = self.linear1(src) # shape(1,6,1024)
        active                  = self.activation(linear1) # shape(1,6,256) #TODO

        src2                    = self.linear2(self.dropout(active)) # shape(1,6,256)
        src                     = src + self.dropout2(src2)
        src                     = self.norm2(src)

        #refining pairwise features
        pairwise_features       = pairwise_features+self.outer_product_mean(src) # calculating pairwise features again after applying multi head attention.
        pairwise_features       = pairwise_features+self.pair_dropout_out(self.triangle_update_out(pairwise_features,src_mask))
        pairwise_features       = pairwise_features+self.pair_dropout_in(self.triangle_update_in(pairwise_features,src_mask))
        pairwise_features       = pairwise_features+self.pair_transition(pairwise_features)

        return src,pairwise_features

class RNAModel(nn.Module):

    def __init__(self, logger, config):
        super(RNAModel, self).__init__()

        self.logger = logger
        self.config = config

        self.config.nhid   = config.ninp * 4

        self.transformer_encoder = []
        self.logger.warning(f"constructing {self.config.nlayers} ConvTransformerEncoderLayers")

        for i in range(self.config.nlayers):
            sub_layer = ConvTransformerEncoderLayer(    
                                d_model                  = self.config.ninp, 
                                nhead                    = self.config.nhead,
                                dim_feedforward          = self.config.nhid, 
                                pairwise_dimension       = self.config.pairwise_dimension,
                                dropout                  = self.config.dropout, 
                                k                        = self.config.k
                            )
            self.transformer_encoder.append(sub_layer)

        self.transformer_encoder     = nn.ModuleList(self.transformer_encoder)
        self.encoder                 = nn.Embedding(self.config.ntoken, self.config.ninp, padding_idx = 4)
        self.decoder                 = nn.Linear(self.config.ninp, self.config.nclass)

        self.outer_product_mean      = OuterProductMean()
        self.pos_encoder             = RelPos(self.config.pairwise_dimension)

    def forward(self, src, src_mask=None):
        
        batch, length = src.shape
        src           = self.encoder(src)
        src           = src.reshape(batch, length, -1)

        pairwise_features = self.outer_product_mean(src)    # Calculating pairwise features using outer product mean. The output shape is (B, L, L, 64) where B is batch size and L is the length of the sequence.  
        pos_encoder       = self.pos_encoder(src)           # Generating positional encoding for the input sequence. The output shape is (B, L, L, 64).
        pairwise_features = pairwise_features + pos_encoder # Adding two metrics of the shape (B, L, L, 64) together.

        for i, layer in enumerate(self.transformer_encoder):
            src, pairwise_features = layer(
                    src, 
                    pairwise_features, 
                    src_mask
                )

        output = self.decoder(src).squeeze(-1)

        return output