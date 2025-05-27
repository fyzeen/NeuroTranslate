import os
import glob
import argparse
#simport yaml
import sys
import math

#import timm #only needed if downloading pretrained models
from datetime import datetime

sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils2.renm_utils import * #load_weights_imagenet
import random

from einops import repeat
from einops.layers.torch import Rearrange

from vit_pytorch.vit import Transformer

from utils import *



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

class FullTransformer(nn.Module): # THIS MODEL CHEATS!
    def __init__(self, dim_model, encoder_depth, nhead, encoder_mlp_dim, decoder_input_dim, decoder_dim_feedforward, decoder_depth, dim_encoder_head, 
                 latent_length=512, num_channels=15, dropout=0.1, num_patches=320, vertices_per_patch=153):
        super(FullTransformer, self).__init__()

        self.dim_model = dim_model
        self.input_dim = decoder_input_dim
        self.latent_length = latent_length

        self.flatten_to_high_dim = nn.Linear(decoder_input_dim, latent_length * dim_model)
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

        self.projection = nn.Linear(latent_length * dim_model, decoder_input_dim)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def encode(self, src):
        return self.encoder(src)
    
    def decode(self, tgt, encoder_out, tgt_mask):
        b, _ = tgt.size()

        # Project to high-dimensional space
        tgt = self.flatten_to_high_dim(tgt)
        tgt = tgt.view(b, -1, self.dim_model)
                
        # Apply positional encoding
        tgt = self.positional_encoding(tgt)

        for layer in self.decoder_layers:
            tgt = layer(tgt=tgt, memory=encoder_out, tgt_mask=tgt_mask)

        tgt = tgt.view(b, -1)
        tgt = self.projection(tgt)

        return torch.tanh(tgt) # returns for ONE cell


    def forward(self, src, tgt, tgt_mask, dropout=0.1):
        b, _ = tgt.size()

        # Project to high-dimensional space
        tgt = self.flatten_to_high_dim(tgt)
        tgt = tgt.view(b, -1, self.dim_model)
        
        # Apply positional encoding
        tgt = self.positional_encoding(tgt)

        encoder_out = self.encoder(src)

        for layer in self.decoder_layers:
            tgt = layer(tgt=tgt, memory=encoder_out, tgt_mask=tgt_mask)

        tgt = tgt.view(b, -1)
        tgt = self.projection(tgt)
        
        return torch.tanh(tgt)

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

        return torch.tanh(tgt) 


    def forward(self, src, tgt, tgt_mask, dropout=0.1):
        b, _ = tgt.size()
        # Project to high-dimensional space
        tgt = self.flatten_to_high_dim(tgt.unsqueeze(-1))
        tgt = tgt.view(b, -1, self.dim_model)
        
        # Apply positional encoding
        tgt = self.positional_encoding(tgt)

        encoder_out = self.encoder(src)

        for layer in self.decoder_layers:
            tgt, masked_attn_weights, cross_attn_weights = layer(tgt=tgt, memory=encoder_out, tgt_mask=tgt_mask)
        
        tgt = tgt.view(b, -1)
        tgt = self.projection(tgt.unsqueeze(-1))
        
        return torch.tanh(tgt.squeeze())


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
                 latent_length=512, num_channels=15, dropout=0.1, num_patches=320, vertices_per_patch=153):
        super(TriuGraphTransformer, self).__init__()

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

        self.projection = MaskedLinear(in_features=latent_length*dim_model, out_features=int((num_out_nodes * (num_out_nodes-1)) / 2), mask=create_mask(num_out_nodes=num_out_nodes, latent_length=latent_length, num_extra_start_tokens=1))

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