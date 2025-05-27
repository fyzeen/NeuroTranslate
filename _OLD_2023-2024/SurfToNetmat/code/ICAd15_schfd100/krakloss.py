import numpy as np
import torch


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