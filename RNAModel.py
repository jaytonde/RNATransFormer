import torch.nn as nn 

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