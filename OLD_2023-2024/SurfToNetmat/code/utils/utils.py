import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

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

def increasing_steps(start, step_sizes):
    current_value = start
    for step in step_sizes:
        yield current_value
        current_value += step

def create_mask(num_out_nodes, latent_length, num_extra_start_tokens):
    '''
    Creates mask for sparse linear layer used in TriuGraphTransformer. This layer will mask (i.e., have a zero) at each position such that the final layer of the 
    TriuGraphTransformer will only connect node n's upper triangle to the representation of node n-1 at the end of the transformer. ** See figure for more details **
    '''
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

def triu_graph_greedy_decode(model, source, latent_length, device, b=1, target=None):
    '''
    Greedy decode algorithm for TriuGraphTransformer. 
    '''
    encoder_output = model.encode(source)
    decoder_input = torch.ones(b, latent_length, latent_length).to(device)
    #decoder_input[:, :, :] = target.to(device)[:, :, :]
    decoder_mask = generate_subsequent_mask(model.latent_length).to(device)

    for i in range(100):
        out = model.decode(encoder_out=encoder_output, tgt=decoder_input, tgt_mask=decoder_mask)
        decoder_input[:, 2:, 2:] = torch.tensor(make_netmat(out))
    
    return decoder_input, out





def xycorr(x,y,axis=1):
    """
    **FROM KRAKENCODER.loss.py**

    Compute correlation between all pairs of rows in x and y (or columns if axis=0)
    
    x: torch tensor or numpy array (Nsubj x M), generally the measured data for N subjects
    y: torch tensor or numpy array (Nsubj x M), generally the predicted data for N subjects
    axis: int (optional, default=1), 1 for row-wise, 0 for column-wise
    
    Returns: torch tensor or numpy array (Nsubj x Nsubj)
    
    NOTE: in train.py we always call cc=xycorr(Ctrue, Cpredicted)
    which means cc[i,:] is cc[true subject i, predicted for all subjects]
    and thus top1acc, which uses argmax(xycorr(true,predicted),axis=1) is:
    for every TRUE output, which subject's PREDICTED output is the best match
    """
    if torch.is_tensor(x):
        cx=x-x.mean(keepdims=True,axis=axis)
        cy=y-y.mean(keepdims=True,axis=axis)
        cx=cx/torch.sqrt(torch.sum(cx ** 2,keepdims=True,axis=axis))
        cy=cy/torch.sqrt(torch.sum(cy ** 2,keepdims=True,axis=axis))
        cc=torch.matmul(cx,cy.t())
    else:
        cx=x-x.mean(keepdims=True,axis=axis)
        cy=y-y.mean(keepdims=True,axis=axis)
        cx=cx/np.sqrt(np.sum(cx ** 2,keepdims=True,axis=axis))
        cy=cy/np.sqrt(np.sum(cy ** 2,keepdims=True,axis=axis))
        cc=np.matmul(cx,cy.T)
    return cc


def correye(x,y):
    """
    **FROM KRAKENCODER.loss.py**

    Loss function: mean squared error between pairwise correlation matrix for xycorr(x,y) and identity matrix
    (i.e., want diagonal to be near 1, off-diagonal to be near 0)
    """
    cc=xycorr(x,y)
    #need keepdim for some reason now that correye and enceye are separated
    loss=torch.norm(cc-torch.eye(cc.shape[0],device=cc.device),keepdim=True)
    return loss

def distance_loss(x,y, margin=None, neighbor=False):
    """
    **FROM KRAKENCODER.loss.py**

    Loss function: difference between self-distance and other-distance for x and y, with optional margin
    If neighbor=True, reconstruction loss applies only to nearest neighbor distance, otherwise to mean distance between all
        off-diagonal pairs.
    
    Inputs:
    x: torch tensor (Nsubj x M), generally the measured data
    y: torch tensor (Nsubj x M), generally the predicted data
    margin: float, optional margin for distance loss (distance above margin is penalized, below is ignored)
    neighbor: bool, (optional, default=False), True for maximizing nearest neighbor distance, False for maximizing mean distance
    
    Returns: 
    loss: torch FloatTensor, difference between self-distance and other-distance
    """
    
    d=torch.cdist(x,y)
    dtrace=torch.trace(d)
    dself=dtrace/d.shape[0] #mean predicted->true distance -- avg distance x_subja to y_subja
    
    if neighbor:
        dnei=d+torch.eye(d.shape[0],device=d.device)*d.max()
        #mean of row-wise min and column-wise min
        dother=torch.mean((dnei.min(axis=0)[0]+dnei.min(axis=1)[0])/2)
    else:
        dother=(torch.sum(d)-dtrace)/(d.shape[0]*(d.shape[0]-1)) #mean predicted->other distance
    
    if margin is not None:
        #dother=torch.min(dother,margin)
        #dother=-torch.nn.ReLU()(dother-margin) #pre 4/5/2024
        #if dother<margin, penalize (lower = more penalty).
        #if dother>=margin, ignore
        #standard triplet loss: torch.nn.ReLU()(dself-dother+margin) or torch.clamp(dself-dother+margin,min=0)
        dother=-torch.nn.ReLU()(margin-dother) #new 4/5/2024
    
    loss=dself-dother
    return loss