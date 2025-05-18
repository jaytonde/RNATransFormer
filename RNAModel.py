import torch
import torch.nn as nn 


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
        self.linear = nn.Linear(17, dim)

    def forward(self, src):
        device      = src.device
        L           = src.shape[1]
        res_id      = torch.arange(L).to(device).unsqueeze(0)           # tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], device='cuda:0')
        bin_values  = torch.arange(-8, 9, device=device)                # tensor([-8, -7, -6, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8], device='cuda:0')
        d           = res_id[:, :, None] - res_id[:, None, :]           # Symmetric metric
        bdy         = torch.tensor(8, device=device)                    # tensor(8, device='cuda:0')
        d           = torch.mininum(torch.maximum(-bdy, d), bdy)        # filtering elements to be the max 8
        d_onehot    = (d[..., None] == bin_values).float()              # Binarizing the array 1 mean there is relation between those neucliotides and 0 means not.

        assert d_onehot.sum(dim=-1).min() == 1
        p           = self.linear(d_onehot)                             # Sends binary matrix to the linar layer which learns relative relationships between neucliotide.
        return p
        

class RNAModel(nn.Module):

    def __init__(self, logger, config):
        super(RNAModel, self).__init__()

        self.logger = logger
        self.config = config

        self.nhid   = config.ninp * 4

        self.transformer_encoder = []
        self.logger(f"constructing {self.config.nlayers} ConvTransformerEncoderLayers")

        for i in range(self.config.nlayers):
            sub_layer = ConvTransformerEncoderLayer
                            (    
                                d_model                  = config.ninp, 
                                nhead                    = config.nhead,
                                dim_feedforward          = nhid, 
                                pairwise_dimension       = config.pairwise_dimension,
                                use_triangular_attention = config.use_triangular_attention,
                                dropout                  = config.dropout, 
                                k                        = k
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

        pairwise_features = self.outer_product_mean(src)
        pairwise_features = pairwise_features + self.pos_encoder(src)

        for i, layer in enumerate(self.transformer_encoder):
            src, pairwise_features = 
            layer(
                    src, 
                    pairwise_features, 
                    src_mask
                )

        output = self.decoder(src).squeeze(-1)

        return output