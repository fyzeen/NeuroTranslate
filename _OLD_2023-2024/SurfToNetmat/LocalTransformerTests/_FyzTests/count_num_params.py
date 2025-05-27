import torch
from models import *

massive_model = FullTransformer(dim_model=192,
                        encoder_depth=6,
                        nhead=6,
                        encoder_mlp_dim=192, 
                        decoder_input_dim=4951, # this is built for an ICA dim _ --> schaefer dim 100 (4951 = n(n-1)/2 + 1 (the +1 is for the start token))
                        decoder_dim_feedforward=192,
                        decoder_depth=6,
                        dim_encoder_head=32, # i fed this in as 32 on accident.. change this back if it starts complaining
                        latent_length=256, # maximum 512 or 256 -- but leads to GINORMOUS projection layers with >100million parameters 
                        dropout=0.1)


large_model = FullTransformer(dim_model=96,
                        encoder_depth=6,
                        nhead=6,
                        encoder_mlp_dim=192, 
                        decoder_input_dim=4951, # this is built for an ICA dim _ --> schaefer dim 100 (4951 = n(n-1)/2 + 1 (the +1 is for the start token))
                        decoder_dim_feedforward=192,
                        decoder_depth=6,
                        dim_encoder_head=32,
                        latent_length=64, # maximum 512 or 256 -- but leads to GINORMOUS projection layers with >100million parameters 
                        dropout=0.1)

small_model = FullTransformer(dim_model=36,
                        encoder_depth=3,
                        nhead=6,
                        encoder_mlp_dim=36, 
                        decoder_input_dim=4951, # this is built for an ICA dim _ --> schaefer dim 100 (4951 = n(n-1)/2 + 1 (the +1 is for the start token))
                        decoder_dim_feedforward=36,
                        decoder_depth=6,
                        dim_encoder_head=6,
                        latent_length=36,
                        dropout=0.1)

massive_model.load_state_dict(torch.load("/home/ahmadf/NeuroTranslate/code/SurfToNetmat/TransformerTest/_FyzTests/TrainedModels/MASSIVE_Model600.pt"))
large_model.load_state_dict(torch.load("/home/ahmadf/NeuroTranslate/code/SurfToNetmat/TransformerTest/_FyzTests/TrainedModels/LARGE_Model600.pt"))
small_model.load_state_dict(torch.load("/home/ahmadf/NeuroTranslate/code/SurfToNetmat/TransformerTest/_FyzTests/TrainedModels/SMALLModel_600.pt"))


massive_model_params = sum(p.numel() for p in massive_model.parameters())
print("Massive model params: ", massive_model_params)

large_model_params = sum(p.numel() for p in large_model.parameters())
print("Massive model params: ", large_model_params)

small_model_params = sum(p.numel() for p in small_model.parameters())
print("Massive model params: ", small_model_params)