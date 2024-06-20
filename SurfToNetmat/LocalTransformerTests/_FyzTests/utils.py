import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

def add_start_token_torch(tensor, n=1, start_value=1):
    """
    Add a new column with a start value to the beginning of each sequence in the input tensor.
    
    :param tensor: Tensor of shape (batch_size, seq_length), input tensor
    :param start_value: int, value to add at the start of each sequence
    :return: Tensor of shape (batch_size, seq_length + 1), tensor with a new column added to the start of each sequence
    """
    batch_size, seq_length = tensor.size()
    new_column = torch.full((batch_size, n), start_value, dtype=tensor.dtype, device=tensor.device)  # Create a new column with the start value
    out = torch.cat([new_column, tensor], dim=1)  # Concatenate the new column with the input tensor
    return out

def add_start_token_np(array, n=1, start_value=1):
    """
    Add a new column with a start value to the beginning of each sequence in the input array.
    
    :param array: Array of shape (batch_size, seq_length), input array
    :param start_value: int, value to add at the start of each sequence
    :return: Array of shape (batch_size, seq_length + 1), array with a new column added to the start of each sequence
    """
    batch_size, seq_length = array.shape
    new_column = np.full((batch_size, n), start_value, dtype=array.dtype)  # Create a new column with the start value
    out = np.concatenate((new_column, array), axis=1)  # Concatenate the new column with the input array
    return out

def write_to_file(content, filepath="/home/ahmadf/batch/sbatch.print", also_print=True):
    with open(filepath, 'a') as file:
        file.write(str(content) + '\n')
    if also_print:
        print(content)
    return
        

def generate_subsequent_mask(size):
    """
    Generate a mask to ensure that each position in the sequence can only attend to
    positions up to and including itself. This is a upper triangular matrix filled with ones.
    
    :param size: int, the length of the sequence
    :return: tensor of shape (size, size), where element (i, j) is False if j <= i, and True otherwise (See attn_mask option here: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
    """
    mask = torch.triu(torch.ones(size, size)).bool()
    mask.diagonal().fill_(False)
    return mask.bool()

def greedy_decode_all(model, source, input_dim, device, b=2): #b=batch size
    '''
    Greedy decode algorithm for a full encoder-decoder architecture (inference).

    Implements using ALL options (initialization with torch.ones vs zeros vs randn; generating ONLY i+1th val VS :i+1 tokens on each iteration)
    '''
    encoder_output = model.encode(source)
    initialization_list = [torch.zeros(b,input_dim).to(device), torch.ones(b,input_dim).to(device), torch.randn(b,input_dim).to(device)]
    # build target mask
    decoder_mask = generate_subsequent_mask(model.latent_length).to(device)

    out_list = []

    for i, decoder_input in enumerate(initialization_list):
        decoder_input_copy = decoder_input.clone()

        for i in range(input_dim-1):
            # compute next output
            out = model.decode(encoder_out=encoder_output, tgt=decoder_input, tgt_mask=decoder_mask)
            decoder_input[:, i+1] = out.squeeze(1)[:, i+1]

        out_list.append(decoder_input.squeeze(0))
        
        for i in range(input_dim-1):
            out = model.decode(encoder_out=encoder_output, tgt=decoder_input_copy, tgt_mask=decoder_mask)
            decoder_input_copy[:, :i+1] = out.squeeze(1)[:, :i+1] 
        
        out_list.append(decoder_input_copy.squeeze(0))

    return out_list

def greedy_decode(model, source, input_dim, device, b):
    '''
    Greedy decode algorithm for a full encoder-decoder architecture (inference).

    Implements uinitialization with torch.ones and generates BOTH i+1th val VS :i+1 tokens on each iteration
    '''
    encoder_output = model.encode(source)
    decoder_input = torch.ones(b,input_dim).to(device)
    # build target mask
    decoder_mask = generate_subsequent_mask(model.latent_length).to(device)

    out_list = []
    decoder_input_copy = decoder_input.clone()

    for i in range(input_dim-1):
        # compute next output
        out = model.decode(encoder_out=encoder_output, tgt=decoder_input, tgt_mask=decoder_mask)
        decoder_input[:, i+1] = out.squeeze(1)[:, i+1]

    out_list.append(decoder_input.squeeze(0))
    
    for i in range(input_dim-1):
        out = model.decode(encoder_out=encoder_output, tgt=decoder_input_copy, tgt_mask=decoder_mask)
        decoder_input_copy[:, :i+1] = out.squeeze(-1)[:, :i+1] 
    
    out_list.append(decoder_input_copy.squeeze(0))

    return out_list

def greedy_decode_conv(model, source, input_dim, device, b):
    '''
    Greedy decode algorithm for a full encoder-decoder architecture (inference) WITH CONV architecture.

    Implements initialization with torch.ones and generates i : i + model.input_dim / model.latent_length on each iteration
    '''
    encoder_output = model.encode(source)
    decoder_input = torch.ones(b,input_dim).to(device)
    # build target mask
    decoder_mask = generate_subsequent_mask(model.latent_length).to(device)

    i=0
    while i < input_dim:
        # compute next output
        out = model.decode(encoder_out=encoder_output, tgt=decoder_input, tgt_mask=decoder_mask)
        decoder_input[:, i:i+int(model.input_dim/model.latent_length)] = out.squeeze(-1)[:, i:i+int(model.input_dim/model.latent_length)]
        i = i + int(model.input_dim/model.latent_length)

    return decoder_input


def make_netmat(data, netmat_dim=100):
    '''
    Makes netmat from upper triangle in numpy
    '''
    sing_sub = int((netmat_dim * (netmat_dim-1))/2)

    # get indeces of upptri cause all these vec netmats are upper trinagles. 
    out_mat_init = np.ones(2*sing_sub+netmat_dim).reshape(netmat_dim,netmat_dim)

    inds_uptri = np.triu_indices_from(out_mat_init,k=1) # k=1 means no diagonal?
    inds_lowtri = np.tril_indices_from(out_mat_init,k=-1) # k=1 means no diagonal?
   
    out_mat_val = out_mat_init
    out_mat_val[inds_lowtri] = data
    out_mat_init[inds_uptri] = out_mat_init.T[inds_uptri]

    return out_mat_init

def make_nemat_allsubj(data, num_nodes):
    '''
    Takes a numpy array of size [num_subj, size_vectorized_netmat] and reshapes to [num_subj, num_nodes, num_nodes]
    '''
    out = np.ones([data.shape[0], num_nodes, num_nodes])
    for i in range(data.shape[0]):
        out[i, :, :] = make_netmat(data[i])
    return out

def add_start_node(data):
    '''
    Takes a numpy array of size [num_subj, num_nodes, num_nodes] and adds a row and column of ones such that the shape becomes [num_subj, num_nodes+1, num_nodes+1]
    '''
    # Get the shape of the input array
    num_subj, num_nodes, _ = data.shape
    
    # Create a new array with increased dimensions
    new_array = np.ones([num_subj, num_nodes + 1, num_nodes + 1])
    
    # Copy the original array into the new array, leaving the last row and column as zeros
    new_array[:, 1:, 1:] = data
    
    return new_array

def greedy_decode_graph(model, source, latent_length, device, b=1):
    '''
    Greedy decode algorithm for a full encoder-decoder architecture (inference) WITH GRAPH architecture.

    Implements initialization with torch.ones [batch_size, num_nodes, num_nodes] and sequentially generates by node
    '''
    encoder_output = model.encode(source)
    decoder_input = torch.ones(b, latent_length, latent_length).to(device)
    # build target mask
    decoder_mask = generate_subsequent_mask(model.latent_length).to(device)

    for i in range(latent_length-1):
        # compute next output
        out = model.decode(encoder_out=encoder_output, tgt=decoder_input, tgt_mask=decoder_mask)
        decoder_input[:, i+1, :] = out[i+1, :]

    return decoder_input


def NEW_greedy_decode_graph(model, source, latent_length, device, b=1):
    '''
    DO NOT USE!
    '''
    encoder_output = model.encode(source)
    decoder_input = torch.ones(b, latent_length, latent_length).to(device)
    # build target mask
    decoder_mask = generate_subsequent_mask(model.latent_length).to(device)

    for i in range(latent_length-1):
        # compute next output
        out = model.decode(encoder_out=encoder_output, tgt=decoder_input, tgt_mask=decoder_mask)
        lower_triangle_col = out[i+2:, i+1]

        # Copy the upper triangle to the corresponding lower triangle
        decoder_input[:, i+1, i+2:] = lower_triangle_col
        decoder_input[:, i+2:, i+1] = lower_triangle_col

    return decoder_input


def increasing_steps(start, step_sizes):
    current_value = start
    for step in step_sizes:
        yield current_value
        current_value += step

def create_mask(num_out_nodes, latent_length, num_extra_start_tokens):
    rowidx = 0
    colidx = num_extra_start_tokens * latent_length

    len_out = (num_out_nodes * (num_out_nodes-1)) / 2
    len_in = latent_length*latent_length

    mask = torch.zeros(int(len_out), int(len_in))

    counter = 1
    for i in increasing_steps(0, range(2, num_out_nodes+1)):
        
        for j in increasing_steps(i, range(counter, num_out_nodes)):
            mask[j, colidx:colidx+latent_length] = 1.0


        colidx = colidx+latent_length
        counter +=1
    
    return mask

def triu_graph_greedy_decode(model, source, latent_length, device, b=1):
    encoder_output = model.encode(source)
    decoder_input = torch.ones(b, latent_length, latent_length).to(device)
    decoder_mask = generate_subsequent_mask(model.latent_length).to(device)

    counter = 1
    for x, i in enumerate(increasing_steps(0, range(2, 100+1))):
        indices = []
        out = model.decode(encoder_out=encoder_output, tgt=decoder_input, tgt_mask=decoder_mask)

        for j in increasing_steps(i, range(counter, 100)):
            indices.append(int(j))

        decoder_input[:, x+2, x+3:] = torch.tensor(out[indices])
        decoder_input[:, x+3:, x+2] = torch.tensor(out[indices])
        

        counter +=1

    return decoder_input, out