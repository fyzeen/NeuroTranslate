import sys
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pytorch.vit import Transformer
from einops.layers.torch import Rearrange

from utils.utils import *

class EncoderSiT(nn.Module):
    def __init__(self, *,
                        dim, 
                        depth,
                        heads,
                        mlp_dim,
                        dim_head,
                        output_length = 512,
                        num_channels = 15,
                        num_patches = 320,
                        num_vertices = 153,
                        dropout = 0.1,
                        emb_dropout = 0.1
                        ):

        super().__init__()
        patch_dim = num_channels * num_vertices

        self.output_length = output_length
        self.dim = dim

        # inputs has size = b * c * n * v where b = batch, c = channels, f = features, n=patches, v=verteces
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout) # See here: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

        self.linear = nn.Linear(num_patches * dim, output_length * dim)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :] # was originally sliced by [:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # Reshape the input tensor to (batch_size, num_patches * dim)
        x_reshaped = x.view(b, -1)
        # Apply the linear layer
        output = self.linear(x_reshaped)
        # Reshape the output tensor to (batch_size, output_length, dim)
        output = output.view(b, self.output_length, self.dim)

        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class TransformerDecoderBlock(nn.Module):
    def __init__(self, input_dim, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.d_model = d_model
        
        self.masked_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Masked Multi-Head Attention
        tgt2, masked_attn_weights = self.masked_multihead_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)  # Residual connection
        tgt = self.norm1(tgt)
        
        # Cross-Multi-Head Attention
        tgt2, cross_attn_weights = self.cross_multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)  # Residual connection
        tgt = self.norm2(tgt)
        
        # Feed Forward
        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(tgt2)  # Residual connection
        tgt = self.norm3(tgt)
        
        return tgt, masked_attn_weights, cross_attn_weights

class ProjectionConvFullTransformer(nn.Module):
    def __init__(self, dim_model, encoder_depth, nhead, encoder_mlp_dim, decoder_input_dim, decoder_dim_feedforward, decoder_depth, dim_encoder_head, 
                 latent_length=512, num_channels=15, dropout=0.1, num_patches=320, vertices_per_patch=153):
        super(ProjectionConvFullTransformer, self).__init__()

        self.dim_model = dim_model
        self.input_dim = decoder_input_dim
        self.latent_length = latent_length

        self.flatten_to_high_dim = nn.Conv1d(in_channels=decoder_input_dim, out_channels=latent_length*dim_model, kernel_size=1, groups=latent_length)
        self.positional_encoding = PositionalEncoding(d_model=dim_model, seq_len=latent_length, dropout=dropout)

        self.encoder = EncoderSiT(dim=dim_model, 
                                  depth=encoder_depth, 
                                  heads=nhead, 
                                  mlp_dim=encoder_mlp_dim,
                                  dim_head=dim_encoder_head,
                                  num_channels=num_channels,  
                                  num_patches=num_patches, 
                                  num_vertices=vertices_per_patch, 
                                  dropout=dropout,
                                  output_length=latent_length,
                                  emb_dropout=0.1)
        
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(input_dim=decoder_input_dim, d_model=dim_model, nhead=nhead, dim_feedforward=decoder_dim_feedforward) for _ in range(decoder_depth)])

        self.projection = nn.Conv1d(in_channels=latent_length*dim_model, out_channels=decoder_input_dim, kernel_size=1, groups=latent_length)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def encode(self, src):
        return self.encoder(src)
    
    def decode(self, tgt, encoder_out, tgt_mask):
        b, _ = tgt.size()

        # Project to high-dimensional space
        tgt = self.flatten_to_high_dim(tgt.unsqueeze(-1))
        tgt = tgt.view(b, -1, self.dim_model)
                
        # Apply positional encoding
        tgt = self.positional_encoding(tgt)

        for layer in self.decoder_layers:
            tgt, masked_attn_weights, cross_attn_weights = layer(tgt=tgt, memory=encoder_out, tgt_mask=tgt_mask)

        tgt = tgt.view(b, -1)
        tgt = self.projection(tgt.unsqueeze(-1))

        return tgt #torch.tanh(tgt) 


    def forward(self, src, tgt, tgt_mask, dropout=0.1):
        b, _ = tgt.size()
        # Project to high-dimensional space
        tgt = self.flatten_to_high_dim(tgt.unsqueeze(-1))
        tgt = tgt.view(b, -1, self.dim_model)
        
        # Apply positional encoding
        tgt = self.positional_encoding(tgt)

        latent = encoder_out = self.encoder(src)

        for layer in self.decoder_layers:
            tgt, masked_attn_weights, cross_attn_weights = layer(tgt=tgt, memory=encoder_out, tgt_mask=tgt_mask)
        
        tgt = latent = tgt.view(b, -1)
        tgt = self.projection(tgt.unsqueeze(-1))
        
        return tgt.squeeze(), latent #torch.tanh(tgt.squeeze()), latent
    
class GraphTransformer(nn.Module):
    def __init__(self, dim_model, encoder_depth, nhead, encoder_mlp_dim, decoder_input_dim, decoder_dim_feedforward, decoder_depth, dim_encoder_head, 
                 latent_length=512, num_channels=15, dropout=0.1, num_patches=320, vertices_per_patch=153):
        super(GraphTransformer, self).__init__()

        self.dim_model = dim_model
        self.input_dim = decoder_input_dim
        self.latent_length = latent_length

        self.encoder = EncoderSiT(dim=dim_model, 
                                  depth=encoder_depth, 
                                  heads=nhead, 
                                  mlp_dim=encoder_mlp_dim,
                                  dim_head=dim_encoder_head,
                                  num_channels=num_channels,  
                                  num_patches=num_patches, 
                                  num_vertices=vertices_per_patch, 
                                  dropout=dropout,
                                  output_length=latent_length,
                                  emb_dropout=0.1)
        
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(input_dim=decoder_input_dim, d_model=dim_model, nhead=nhead, dim_feedforward=decoder_dim_feedforward) for _ in range(decoder_depth)])

        self.projection = nn.Conv1d(in_channels=latent_length*dim_model, out_channels=latent_length*dim_model, kernel_size=1, groups=latent_length)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src):
        return self.encoder(src)
    
    def decode(self, tgt, encoder_out, tgt_mask):
        b, _, _ = tgt.size()

        tgt = tgt.view(b, self.latent_length, self.dim_model)

        for layer in self.decoder_layers:
            tgt, masked_attn_weights, cross_attn_weights = layer(tgt=tgt, memory=encoder_out, tgt_mask=tgt_mask)

        tgt = tgt.transpose(1, 2).reshape(b, -1)
        tgt = self.projection(tgt.unsqueeze(-1))
        tgt = tgt.view(b, self.latent_length, self.dim_model).transpose(2,1)

        return torch.tanh(tgt.squeeze()) 


    def forward(self, src, tgt, tgt_mask, dropout=0.1):
        b, _, _ = tgt.size()

        tgt = tgt.view(b, self.latent_length, self.dim_model)

        encoder_out = self.encoder(src)

        for layer in self.decoder_layers:
            tgt, masked_attn_weights, cross_attn_weights = layer(tgt=tgt, memory=encoder_out, tgt_mask=tgt_mask)

        tgt = tgt.transpose(1, 2).reshape(b, -1)
        tgt = self.projection(tgt.unsqueeze(-1))
        tgt = tgt.view(b, self.latent_length, self.dim_model).transpose(2,1)

        return torch.tanh(tgt.squeeze())
    

class MaskedLinear(nn.Linear):
    def __init__(self, *args, mask, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask

    def forward(self, input):
        return F.linear(input, self.weight*self.mask, self.bias)


class TriuGraphTransformer(nn.Module):
    def __init__(self, dim_model, encoder_depth, nhead, encoder_mlp_dim, decoder_input_dim, decoder_dim_feedforward, decoder_depth, dim_encoder_head, num_out_nodes=100,
                 latent_length=512, num_channels=15, dropout=0.1, num_patches=320, vertices_per_patch=153, extra_start_tokens=1):
        super(TriuGraphTransformer, self).__init__()

        self.dim_model = dim_model
        self.input_dim = decoder_input_dim
        self.latent_length = latent_length
        self.extra_start_tokens = extra_start_tokens

        self.encoder = EncoderSiT(dim=dim_model, 
                                  depth=encoder_depth, 
                                  heads=nhead, 
                                  mlp_dim=encoder_mlp_dim,
                                  dim_head=dim_encoder_head,
                                  num_channels=num_channels,  
                                  num_patches=num_patches, 
                                  num_vertices=vertices_per_patch, 
                                  dropout=dropout,
                                  output_length=latent_length,
                                  emb_dropout=0.1)
        
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(input_dim=decoder_input_dim, d_model=dim_model, nhead=nhead, dim_feedforward=decoder_dim_feedforward) for _ in range(decoder_depth)])

        self.projection = MaskedLinear(in_features=latent_length*dim_model, out_features=int((num_out_nodes * (num_out_nodes-1)) / 2), mask=create_mask(num_out_nodes=num_out_nodes, latent_length=latent_length, num_extra_start_tokens=extra_start_tokens))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src):
        return self.encoder(src)
    
    def decode(self, tgt, encoder_out, tgt_mask):
        b, _, _ = tgt.size()

        tgt = tgt.view(b, self.latent_length, self.dim_model)

        for layer in self.decoder_layers:
            tgt, masked_attn_weights, cross_attn_weights = layer(tgt=tgt, memory=encoder_out, tgt_mask=tgt_mask)

        tgt = tgt.transpose(1, 2).reshape(b, -1)
        tgt = self.projection(tgt)

        return torch.tanh(tgt.squeeze()) 


    def forward(self, src, tgt, tgt_mask, dropout=0.1):
        b, _, _ = tgt.size()

        tgt = tgt.view(b, self.latent_length, self.dim_model)

        encoder_out = self.encoder(src)

        for layer in self.decoder_layers:
            tgt, masked_attn_weights, cross_attn_weights = layer(tgt=tgt, memory=encoder_out, tgt_mask=tgt_mask)

        tgt = tgt.transpose(1, 2).reshape(b, -1)
        tgt = self.projection(tgt)

        return torch.tanh(tgt.squeeze())

# From Samuel below:
class SiT_nopool_linout(nn.Module):
    def __init__(self, *,
                        dim, 
                        depth,
                        heads,
                        mlp_dim,
                        num_patches = 320,
                        num_classes= 4950,
                        num_channels = 15,
                        num_vertices = 153,
                        dim_head = 64,
                        dropout = 0.3,
                        emb_dropout = 0.1
                        ):

        super().__init__()

        # features of maps only add to number of patches, but dim of each patch stays at 4*153
        patch_dim = num_channels * num_vertices # flattened patch

        # linear embedding of the vectorized patches
        # inputs has size = b * c * n * v, I think this changes depending on input featues for meshes, so 
        # size = b * c * f * n * v where b = batch c = channels f = features, n=patches?, v=verteces?
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        ) # linear layer embeds the inputdim*153 -> attention dim

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim)) # plus one for regressino token according to paper
        # good guide I used to walkthrough: https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout) # torch transformer

        decomp_attnpatch = num_patches * dim # size of decomposed patch and their attention vector

        self.rearrange = Rearrange('b n d  -> b (n d)') # decomp them here, which will be the size of decomp_attnpatch

        self.linear = nn.Linear(decomp_attnpatch,num_classes) # linear project from batch x 122k -> batch 4950

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img):
        #write_to_file('Looking into vectorized brain maps shape: {}'.format(img.size())) # should have size batch, chan, sphere(s) count, patches, verteces  
        x = self.to_patch_embedding(img)
        _, n, _ = x.shape # look above at to_patch_embedding see that it 'b c n v  -> b n (v c)' 256x320x384 if training at least
        #write_to_file('Performed Patch embedding, and has shape:{}'.format(x.shape))

        x += self.pos_embedding[:,:(n)] # spatial relationship across patches based on pos embedding of tokens
        #write_to_file('Performed pos emb, now has shape: {}'.format(x.shape))
        
        x = self.dropout(x)
        #write_to_file('Dropout used, now has shape: {}'.format(x.shape))
        
        x = self.transformer(x) # give embedded input to transformer architecture
        #write_to_file('Passed through transformer architecture, now has shape: {}'.format(x.shape))

        x = latent = self.rearrange(x)

        x = self.linear(x)
        #write_to_file('Collapsed patchesxattndim, now projected linearly to num_classes - has shape: {}'.format(x.shape))
        
        return x, latent
    
class VariationalSiT_nopool_linout(nn.Module):
    def __init__(self, *,
                        dim, 
                        depth,
                        heads,
                        mlp_dim,
                        VAE_latent_dim = 500,
                        num_patches = 320,
                        num_classes= 4950,
                        num_channels = 15,
                        num_vertices = 153,
                        dim_head = 64,
                        dropout = 0.3,
                        emb_dropout = 0.1
                        ):

        super().__init__()

        # features of maps only add to number of patches, but dim of each patch stays at 4*153
        patch_dim = num_channels * num_vertices # flattened patch

        # linear embedding of the vectorized patches
        # inputs has size = b * c * n * v, I think this changes depending on input featues for meshes, so 
        # size = b * c * f * n * v where b = batch c = channels f = features, n=patches?, v=verteces?
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        ) # linear layer embeds the inputdim*153 -> attention dim

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim)) 
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout) # torch transformer

        decomp_attnpatch = num_patches * dim # size of decomposed patch and their attention vector

        self.rearrange = Rearrange('b n d  -> b (n d)') # decomp them here, which will be the size of decomp_attnpatch

        self.fc_mu = nn.Linear(decomp_attnpatch, VAE_latent_dim) # linear project from batch x 122k -> batch 500
        self.fc_var = nn.Linear(decomp_attnpatch, VAE_latent_dim)
        
        self.projection = nn.Linear(VAE_latent_dim, num_classes)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img):
        #write_to_file('Looking into vectorized brain maps shape: {}'.format(img.size())) # should have size batch, chan, sphere(s) count, patches, verteces  
        x = self.to_patch_embedding(img)
        _, n, _ = x.shape # look above at to_patch_embedding see that it 'b c n v  -> b n (v c)' 256x320x384 if training at least
        #write_to_file('Performed Patch embedding, and has shape:{}'.format(x.shape))

        x += self.pos_embedding[:,:(n)] # spatial relationship across patches based on pos embedding of tokens
        #write_to_file('Performed pos emb, now has shape: {}'.format(x.shape))
        
        x = self.dropout(x)
        #write_to_file('Dropout used, now has shape: {}'.format(x.shape))
        
        x = self.transformer(x) # give embedded input to transformer architecture
        #write_to_file('Passed through transformer architecture, now has shape: {}'.format(x.shape))

        x = self.rearrange(x)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        x = mu + (std * epsilon)

        x = self.projection(x)
        
        return x, mu, log_var

class VariationalConvTransformer(nn.Module):
    def __init__(self, dim_model, encoder_depth, nhead, encoder_mlp_dim, decoder_input_dim, decoder_dim_feedforward, decoder_depth, dim_encoder_head, 
                 VAE_latent_dim=1000, latent_length=512, num_channels=15, dropout=0.1, num_patches=320, vertices_per_patch=153):
        super(VariationalConvTransformer, self).__init__()

        self.dim_model = dim_model
        self.input_dim = decoder_input_dim
        self.latent_length = latent_length

        self.flatten_to_high_dim = nn.Conv1d(in_channels=decoder_input_dim, out_channels=latent_length*dim_model, kernel_size=1, groups=latent_length)
        self.positional_encoding = PositionalEncoding(d_model=dim_model, seq_len=latent_length, dropout=dropout)

        self.encoder = EncoderSiT(dim=dim_model, 
                                  depth=encoder_depth, 
                                  heads=nhead, 
                                  mlp_dim=encoder_mlp_dim,
                                  dim_head=dim_encoder_head,
                                  num_channels=num_channels,  
                                  num_patches=num_patches, 
                                  num_vertices=vertices_per_patch, 
                                  dropout=dropout,
                                  output_length=latent_length,
                                  emb_dropout=0.1)
        
        self.fc_mu = nn.Linear(dim_model * latent_length, VAE_latent_dim)
        self.fc_var = nn.Linear(dim_model * latent_length, VAE_latent_dim)

        self.vae_latent_to_encoder_out = nn.Linear(VAE_latent_dim, dim_model * latent_length)
        
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(input_dim=decoder_input_dim, d_model=dim_model, nhead=nhead, dim_feedforward=decoder_dim_feedforward) for _ in range(decoder_depth)])

        self.projection = nn.Conv1d(in_channels=latent_length*dim_model, out_channels=decoder_input_dim, kernel_size=1, groups=latent_length)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def encode(self, src):
        x = self.encoder(src)
        x = x.view(x.size()[0], -1) # reshape to [b x model_dim * latent_length]

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]
    
    def decode(self, tgt, encoder_out, tgt_mask):
        b, _ = tgt.size()

        # Project to high-dimensional space
        tgt = self.flatten_to_high_dim(tgt.unsqueeze(-1))
        tgt = tgt.view(b, -1, self.dim_model)
                
        # Apply positional encoding
        tgt = self.positional_encoding(tgt)

        # Reparameterization trick to sample from latent space
        mu = encoder_out[0]
        log_var = encoder_out[1]
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + (std * epsilon)

        vae_in_encoder_space = self.vae_latent_to_encoder_out(z)
        vae_in_encoder_space = vae_in_encoder_space.view(b, self.latent_length, self.dim_model)


        for layer in self.decoder_layers:
            tgt, masked_attn_weights, cross_attn_weights = layer(tgt=tgt, memory=vae_in_encoder_space, tgt_mask=tgt_mask)

        tgt = tgt.view(b, -1)
        tgt = self.projection(tgt.unsqueeze(-1))

        return tgt #torch.tanh(tgt) 


    def forward(self, src, tgt, tgt_mask, dropout=0.1):
        b, _ = tgt.size()
        # Project to high-dimensional space
        tgt = self.flatten_to_high_dim(tgt.unsqueeze(-1))
        tgt = tgt.view(b, -1, self.dim_model)
        
        # Apply positional encoding
        tgt = self.positional_encoding(tgt)

        encoder_out = self.encode(src)
        
        # Reparameterization trick to sample from latent space
        mu = encoder_out[0]
        log_var = encoder_out[1]
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + (std * epsilon)

        vae_in_encoder_space = self.vae_latent_to_encoder_out(z)
        vae_in_encoder_space = vae_in_encoder_space.view(b, self.latent_length, self.dim_model)


        for layer in self.decoder_layers:
            tgt, masked_attn_weights, cross_attn_weights = layer(tgt=tgt, memory=vae_in_encoder_space, tgt_mask=tgt_mask)
        
        tgt = tgt.view(b, -1)
        tgt = self.projection(tgt.unsqueeze(-1))
        
        return tgt.squeeze(), mu, log_var #torch.tanh(tgt.squeeze()), mu, log_var
    

class TwoHemi_SiT_nopool_linout(nn.Module):
    def __init__(self, *,
                        dim, 
                        depth,
                        heads,
                        mlp_dim,
                        latent_dim = 500,
                        num_patches = 320,
                        num_classes= 4950,
                        num_channels = 15,
                        num_vertices = 153,
                        dim_head = 64,
                        dropout = 0.3,
                        emb_dropout = 0.1
                        ):

        super().__init__()

        patch_dim = num_channels * num_vertices # flattened patch

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.elu  = nn.ELU()

        # HEMI 1
        self.hemi1_to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        ) 

        self.hemi1_pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.hemi1_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout) 

        decomp_attnpatch = num_patches * dim 

        self.rearrange = Rearrange('b n d  -> b (n d)') 

        self.hemi1_linear = nn.Linear(decomp_attnpatch, latent_dim) # linear project from batch x 122k -> batch x latent_dim

        # HEMI 2
        self.hemi2_to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        ) 

        self.hemi2_pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.hemi2_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout) 

        self.hemi2_linear = nn.Linear(decomp_attnpatch, latent_dim) # linear project from batch x 122k -> batch x latent_dim

        # share hemi information and project --- !IMPORTANT! FOR "LARGE" (not "LARGER" models), output of sharehemis should be `int(num_classes/2)`
        self.share_hemis = nn.Linear(latent_dim*2, int(num_classes/2))
        self.project = nn.Linear(int(num_classes/2), int(num_classes))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, hemi1, hemi2):

        x = self.hemi1_to_patch_embedding(hemi1)
        _, n, _ = x.shape 
        x += self.hemi1_pos_embedding[:,:(n)]       
        x = self.dropout(x)        
        x = self.hemi1_transformer(x)
        x = self.rearrange(x)
        x = self.hemi1_linear(x) 
        hemi1_latent = x#self.relu(x)

        x = self.hemi2_to_patch_embedding(hemi2)
        _, n, _ = x.shape 
        x += self.hemi2_pos_embedding[:,:(n)]       
        x = self.dropout(x)        
        x = self.hemi2_transformer(x)
        x = self.rearrange(x)
        x = self.hemi2_linear(x) 
        hemi2_latent = x#self.relu(x)
        
        x = torch.cat((hemi1_latent, hemi2_latent), dim=1)
        x = latent = self.share_hemis(x)
        #x = self.relu(x)

        x = self.project(x)

        return x, latent
    

class NetmatEncoder(nn.Module):
    def __init__(self, 
                 input_dim,
                 dim_model, 
                 nhead, 
                 num_layers, 
                 dim_feedforward,
                 output_length, 
                 dropout=0.1):
        
        super(NetmatEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_length = output_length
        self.dim_model = dim_model

        self.positional_encoding = PositionalEncoding(d_model=dim_model, seq_len=input_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.linear = nn.Linear(input_dim * dim_model, output_length * dim_model)
        
    def forward(self, src):
        x = self.positional_encoding(src)
        x = self.transformer_encoder(src)

        b, seq_len, dim_model = x.size()

        x = x.view(b, seq_len * dim_model)
        x = self.linear(x)
        x = x.view(b, self.output_length, self.dim_model)

        return x
    

class ProjectionConvTransformerNetmat(nn.Module):
    def __init__(self, dim_model, encoder_input_dim, encoder_model_len, encoder_depth, nhead_encoder, encoder_mlp_dim, decoder_input_dim, nhead_decoder, decoder_dim_feedforward, decoder_depth, 
                 latent_length=512, dropout=0.1):
        super(ProjectionConvTransformerNetmat, self).__init__()

        self.dim_model = dim_model
        self.input_dim = decoder_input_dim
        self.latent_length = latent_length

        # encoder
        #self.flatten_to_high_dim_encoder = nn.Linear(encoder_input_dim, encoder_model_len*dim_model) #nn.Conv1d(in_channels=encoder_input_dim, out_channels=encoder_model_len*dim_model, kernel_size=1, groups=encoder_model_len)
        self.flatten_to_high_dim_encoder = nn.Conv1d(in_channels=encoder_input_dim, out_channels=encoder_model_len*dim_model, kernel_size=1, groups=encoder_model_len)
        self.encoder = NetmatEncoder(input_dim = encoder_model_len,
                                     dim_model = dim_model, 
                                     nhead = nhead_encoder, 
                                     num_layers = encoder_depth, 
                                     dim_feedforward = encoder_mlp_dim,
                                     output_length = latent_length)
        
        # decoder
        self.flatten_to_high_dim_decoder = nn.Conv1d(in_channels=decoder_input_dim, out_channels=latent_length*dim_model, kernel_size=1, groups=latent_length)
        self.positional_encoding_decoder = PositionalEncoding(d_model=dim_model, seq_len=latent_length, dropout=dropout)
        
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(input_dim=decoder_input_dim, d_model=dim_model, nhead=nhead_decoder, dim_feedforward=decoder_dim_feedforward) for _ in range(decoder_depth)])

        self.projection = nn.Conv1d(in_channels=latent_length*dim_model, out_channels=decoder_input_dim, kernel_size=1, groups=latent_length)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def encode(self, src):
        b, _ = src.size()

        # Project to high-dimensional space
        x = self.flatten_to_high_dim_encoder(src.unsqueeze(-1)) # need .unsqueeze(-1) if using sparse
        x = x.view(b, -1, self.dim_model)
        
        return self.encoder(x)
    
    def decode(self, tgt, encoder_out, tgt_mask):
        b, _ = tgt.size()

        # Project to high-dimensional space
        tgt = self.flatten_to_high_dim_decoder(tgt.unsqueeze(-1))
        tgt = tgt.view(b, -1, self.dim_model)
                
        # Apply positional encoding
        tgt = self.positional_encoding_decoder(tgt)

        for layer in self.decoder_layers:
            tgt, masked_attn_weights, cross_attn_weights = layer(tgt=tgt, memory=encoder_out, tgt_mask=tgt_mask)

        tgt = tgt.view(b, -1)
        tgt = self.projection(tgt.unsqueeze(-1))

        return tgt #torch.tanh(tgt) 


    def forward(self, src, tgt, tgt_mask, dropout=0.1):
        b, _ = tgt.size()
        # Project to high-dimensional space
        tgt = self.flatten_to_high_dim_decoder(tgt.unsqueeze(-1))
        tgt = tgt.view(b, -1, self.dim_model)
        
        # Apply positional encoding
        tgt = self.positional_encoding_decoder(tgt)

        latent = encoder_out = self.encode(src)

        for layer in self.decoder_layers:
            tgt, masked_attn_weights, cross_attn_weights = layer(tgt=tgt, memory=encoder_out, tgt_mask=tgt_mask)
        
        tgt = tgt.view(b, -1)
        tgt = self.projection(tgt.unsqueeze(-1))
        
        return tgt.squeeze(), latent 
    

class TransformerEncoder_Netmat(nn.Module):
    def __init__(self, dim_model, input_dim, model_len, depth, nhead, mlp_dim, output_len,
                 latent_length=512, dropout=0.1):

        super().__init__()

        self.dim_model = dim_model
        self.input_dim = input_dim
        self.latent_length = latent_length

        # encoder
        #self.flatten_to_high_dim_encoder = nn.Linear(input_dim, model_len*dim_model) 
        self.flatten_to_high_dim_encoder = nn.Conv1d(in_channels=input_dim, out_channels=model_len*dim_model, kernel_size=1, groups=model_len)

        self.encoder = NetmatEncoder(input_dim = model_len,
                                     dim_model = dim_model, 
                                     nhead = nhead, 
                                     num_layers = depth, 
                                     dim_feedforward = mlp_dim,
                                     output_length = latent_length)
        


        self.linear = nn.Linear(latent_length*dim_model, output_len) 

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img):
        b, _ = img.size()

        # Project to high-dimensional space
        x = self.flatten_to_high_dim_encoder(img.unsqueeze(-1)) # need .unsqueeze(-1) if using sparse
        x = x.view(b, -1, self.dim_model)
        x = self.encoder(x)

        x = latent = x.view(b, -1)

        x = self.linear(x)

        return x, latent
    
class VariationalTransformerEncoder_Netmat(nn.Module):
    def __init__(self, dim_model, input_dim, model_len, depth, nhead, mlp_dim, output_len, VAE_latent_dim,
                 latent_length=512, dropout=0.1):

        super().__init__()

        self.dim_model = dim_model
        self.input_dim = input_dim
        self.latent_length = latent_length

        # encoder
        #self.flatten_to_high_dim_encoder = nn.Linear(input_dim, model_len*dim_model)
        self.flatten_to_high_dim_encoder = nn.Conv1d(in_channels=input_dim, out_channels=model_len*dim_model, kernel_size=1, groups=model_len)

        self.encoder = NetmatEncoder(input_dim = model_len,
                                     dim_model = dim_model, 
                                     nhead = nhead, 
                                     num_layers = depth, 
                                     dim_feedforward = mlp_dim,
                                     output_length = latent_length)
        

        self.fc_mu = nn.Linear(latent_length*dim_model, VAE_latent_dim) # linear project from batch x 122k -> batch 500
        self.fc_var = nn.Linear(latent_length*dim_model, VAE_latent_dim)
        
        self.projection = nn.Linear(VAE_latent_dim, output_len)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img):
        b, _ = img.size()

        # Project to high-dimensional space
        x = self.flatten_to_high_dim_encoder(img.unsqueeze(-1)) # need .unsqueeze(-1) if using sparse
        x = x.view(b, -1, self.dim_model)
        x = self.encoder(x)

        x = x.view(b, -1)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        x = mu + (std * epsilon)

        x = self.projection(x)

        return x, mu, log_var