# -*- coding: utf-8 -*-
"""
@file train_utils.py
@brief Utility functions and classes for training the NSTSC model.

Created on Tue Oct 11 10:30:37 2022
@author: yanru
"""

import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from Models_node import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Node:
    """
    @class Node
    @brief Represents a node in the NSTSC tree.
    """
    def __init__(self, nodei):
        """
        @brief Initialize a Node with an index.
        @param nodei Index of the node.
        """
        self.idx = nodei


# Assign training data to a node
def Givetraintonode(Nodes, pronodenum, datanums):
    # print("Running Givetraintonode")
    """
    @brief Assigns training data indices to a node.
    @param Nodes Dictionary of nodes.
    @param pronodenum Node index to assign data to.
    @param datanums List of data indices.
    @return Updated Nodes dictionary.
    """
    Nodes[pronodenum].trainidx = datanums
    return Nodes


# Assign validation data to a node
def Givevaltonode(Nodes, pronodenum, datanums):
    # print("Running Givevaltonode")
    """
    @brief Assigns validation data indices to a node.
    @param Nodes Dictionary of nodes.
    @param pronodenum Node index to assign data to.
    @param datanums List of data indices.
    @return Updated Nodes dictionary.
    """
    Nodes[pronodenum].testidx = datanums
    return Nodes


# Train a NSTSC model given training data
def Train_model(Xtrain_raw, Xval_raw, ytrain_raw, yval_raw, epochs=100, normalize_timeseries=True, lr=0.1):
    # print("Running Train_model")
    """
    @brief Train a NSTSC model given training and validation data.
    @param Xtrain_raw Training data features.
    @param Xval_raw Validation data features.
    @param ytrain_raw Training data labels.
    @param yval_raw Validation data labels.
    @param epochs Number of training epochs.
    @param normalize_timeseries Whether to normalize the time series.
    @param lr Learning rate.
    @return Trained tree model.
    """
    # classnum = int(np.max(ytrain_raw) + 1)
    #Code for when ytrain_raw is on GPU
    classnum = int(torch.max(ytrain_raw).item() + 1)
    Tree = Build_tree(Xtrain_raw, Xval_raw, ytrain_raw, yval_raw, epochs, classnum, learnrate=lr, savepath='./utils/')
    Tree = Prune_tree(Tree, Xval_raw, yval_raw)
    return Tree


# Construct a tree from node phase classifiers
def Build_tree(Xtrain, Xval, ytrain_raw, yval_raw, Epoch, classnum, learnrate, savepath="./utils/"):
    # print("Running Build_tree")
    """
    @brief Construct a tree from node phase classifiers.
    @param Xtrain: Training data features.
    @param Xval: Validation data features.
    @param ytrain_raw: Training data labels.
    @param yval_raw: Validation data labels.
    @param Epoch: Number of training epochs.
    @param classnum: Number of classes.
    @param learnrate: Learning rate for training.
    @param savepath: Path to save models.
    @return Tree dictionary.
    """
    Tree = {}
    pronodenum = 0
    maxnodenum = 0
    Modelnum = 7
    bstaccu = 0
    Tree[maxnodenum] = Node(maxnodenum)
    Tree[pronodenum].stoptrain = False
    Tree = Givetraintonode(Tree, pronodenum, list(range(len(ytrain_raw))))
    Tree = Givevaltonode(Tree, pronodenum, list(range(len(yval_raw))))
    while pronodenum <= maxnodenum:
        if not Tree[pronodenum].stoptrain:
            Tree, trueidx, falseidx, trueidxt,\
            falseidxt, = Trainnode(Tree, pronodenum, Epoch, learnrate,\
            Xtrain, ytrain_raw, Modelnum, savepath, classnum,\
                Xval, yval_raw)
        
            if maxnodenum < 128:
                if len(Tree[pronodenum].trueidx) > 0:
                    Tree, maxnodenum = Updateleftchd(Tree, pronodenum,\
                    maxnodenum, Xtrain, ytrain_raw, classnum,\
                        Xval, yval_raw)
                if len(Tree[pronodenum].falseidx) > 0:
                    Tree, maxnodenum = Updaterigtchd(Tree, pronodenum,\
                    maxnodenum, Xtrain, ytrain_raw, classnum,\
                        Xval, yval_raw)
    
        pronodenum += 1


    return Tree


# Train a node phase classifier
def Trainnode(Nodes, pronum, Epoch, lrt, X, y, Mdlnum, mdlpath, clsnum, Xt, yt):
    # print("Running Trainnode")
    """
    @brief Train a node phase classifier.
    @param Nodes: Dictionary of nodes.
    @param pronum: Current node index.
    @param Epoch: Number of epochs.
    @param lrt: Learning rate.
    @param X: Training features.
    @param y: Training labels.
    @param Mdlnum: Model number.
    @param mdlpath: Path to save model.
    @param clsnum: Number of classes.
    @param Xt: Validation features.
    @param yt: Validation labels.
    @return Updated Nodes and split indices.
    """
    trainidx = Nodes[pronum].trainidx    
    Xori = X[trainidx,:]
    yori = y[trainidx]
    testidx = Nodes[pronum].testidx
    Xorit = Xt[testidx,:]
    yorit = yt[testidx]
    yoricount = County(yori, clsnum)
    yoricountt = County(yorit, clsnum)
    # curclasses = np.where(yoricount!=0)[0]

    #code for when yori is on GPU
    curclasses = torch.where(yoricount!=0)[0]
    Nodes[pronum].ycount = yoricount
    Nodes[pronum].predcls = yoricountt.argmax()
    yecds = Ecdlabel(yori, curclasses)
    yecdst = Ecdlabel(yorit, curclasses)
    # yori = np.array(yori)
    # yori = torch.LongTensor(yori).to(device) #move it to GPU
    # yorit = torch.LongTensor(yorit).to(device) #move it to GPU
    N, T = len(yori), int(Xori.shape[1]/3)
    ginibest = 10
    # Xori = torch.Tensor(Xori).to(device) #move it to GPU
    # Xorit = torch.Tensor(Xorit).to(device) #move it to GPU
    batch_size = N // 20
    if batch_size <= 1:
        batch_size = N
        
    for mdlnum in range(1, Mdlnum):
        # print(f"Model Number: {mdlnum}")
        tlnns = {}
        optimizers = {}
        X_rns = {}
        Losses = {}
        for i in curclasses:
            i_int = int(i.item())  # Convert tensor to Python integer
            # Move the model and input tensors to the GPU device
            tlnns[i_int] = eval('TL_NN' + str(mdlnum) + '(T)').to(device)
            optimizers[i_int] = torch.optim.AdamW(tlnns[i_int].parameters(), lr = lrt)
        
        ginisall = []
        for epoch in range(Epoch):
            # print(f"Epoch: {epoch}")        
            for d_i in range(N//batch_size + 1):
                rand_idx = np.array(range(d_i*batch_size, min((d_i+1)*batch_size,\
                                N)))           
                for Ci in curclasses:
                    Ci_int = int(Ci.item())  # Convert tensor to Python integer
                    ytrain = yecds[Ci_int]
                    ytest = yecdst[Ci_int]
                    IR = sum(ytrain==1)/sum(ytrain==0) 
                    # ytrain = torch.LongTensor(ytrain).to(device) #move it to GPU
                    # ytest = torch.LongTensor(ytest).to(device) #move it to GPU
                    X_batch = torch.Tensor(Xori[rand_idx,:])
                    y_batch = ytrain[rand_idx]
                    w_batch = IR * (1-y_batch) 
                    w_batch[w_batch==0] = 1
                    X_rns[Ci_int] = tlnns[Ci_int](X_batch[:,:T], X_batch[:,T:2*T],\
                                          X_batch[:,2*T:])
                    Losses[Ci_int] =  torch.sum(w_batch * (-y_batch * \
                                  torch.log(X_rns[Ci_int] + 1e-9) - (1-y_batch) * \
                                  torch.log(1-X_rns[Ci_int] + 1e-9)))
                    
                    optimizers[Ci_int].zero_grad()
                    Losses[Ci_int].backward()
                    optimizers[Ci_int].step()
                
                if d_i % 10 == 0:
                    giniscores = torch.Tensor(Cptginisplit(tlnns, Xorit, yorit,\
                                                           T, clsnum))
                    ginisminnum = int(giniscores.argmin().numpy())
                    ginismin = giniscores.min()
                    ginisall.append(ginismin)
                    if ginismin < ginibest:
                        best_class = int(curclasses[ginisminnum].item())  # Convert to Python integer
                        torch.save(tlnns[best_class], mdlpath + 'bestmodel.pkl')
                        # Nodes[pronum].predcls = ginisminnum
                        Nodes[pronum].ginis = ginismin
                        ginibest = ginismin
                        Nodes[pronum].bstmdlclass = best_class
                    
                    
    Nodes[pronum].bestmodel = torch.load(mdlpath + 'bestmodel.pkl', weights_only=False)
                 
    Xpred, accu, trueidx, falseidx = Cpt_Accuracy(Nodes[pronum].bestmodel,\
                                Xori, yecds[Nodes[pronum].bstmdlclass], T)
    Xpredt, accut, trueidxt, falseidxt = Cpt_Accuracy(Nodes[pronum].bestmodel,\
                                Xorit, yecdst[Nodes[pronum].bstmdlclass], T)
        
    
    # Nodes[pronum].trueidx = np.array(Nodes[pronum].trainidx)[trueidx]
    # Nodes[pronum].falseidx = np.array(Nodes[pronum].trainidx)[falseidx]
    # Nodes[pronum].trueidxt = np.array(Nodes[pronum].testidx)[trueidxt]
    # Nodes[pronum].falseidxt = np.array(Nodes[pronum].testidx)[falseidxt]

    # Convert trueidx and falseidx to a list of integers - numpy doesn't support tensor based indexing
    Nodes[pronum].trueidx = [Nodes[pronum].trainidx[i.item()] for i in trueidx]
    Nodes[pronum].falseidx = [Nodes[pronum].trainidx[i.item()] for i in falseidx]
    Nodes[pronum].trueidxt = [Nodes[pronum].testidx[i.item()] for i in trueidxt]
    Nodes[pronum].falseidxt = [Nodes[pronum].testidx[i.item()] for i in falseidxt]
    return Nodes, trueidx, falseidx, trueidxt, falseidxt    


# Expand left child node
def Updateleftchd(Nodes, pronum, maxnum, Xori, yori, clsnum, Xorit, yorit):
    # print("Running Updateleftchd")
    """
    @brief Expand left child node.
    @param Nodes: Dictionary of nodes.
    @param pronum: Parent node index.
    @param maxnum: Maximum node index.
    @param Xori: Training features.
    @param yori: Training labels.
    @param clsnum: Number of classes.
    @param Xorit: Validation features.
    @param yorit: Validation labels.
    @return Updated Nodes and maxnum.
    """
    Leftidx = Nodes[pronum].trueidx
    Leftidxt = Nodes[pronum].trueidxt
    yleft = yori[Leftidx]
    ylgini = Cptgininode(yleft, clsnum)
    yleftt = yorit[Leftidxt]
    ylginit = Cptgininode(yleftt, clsnum)
    maxnum += 1
    Nodes[maxnum] = Node(maxnum)
    Nodes = Givetraintonode(Nodes, maxnum, Leftidx)
    Nodes = Givevaltonode(Nodes, maxnum, Leftidxt)
    ylcount = County(yleft, clsnum)
    ylcountt = County(yleftt, clsnum)
    Nodes[maxnum].ycount = ylcount
    Nodes[maxnum].ycountt = ylcountt
    Nodes[maxnum].predcls = ylcountt.argmax()
    Nodes[maxnum].ginis = ylgini
    Nodes[maxnum].ginist = ylginit
    
    if ylginit == 0 or ylgini == 0:
        Nodes[maxnum].stoptrain = True
    else:
        Nodes[maxnum].stoptrain = False
    Nodes[pronum].leftchd = maxnum
    Nodes[maxnum].prntnb = pronum
    Nodes[maxnum].childtype = 'leftchild'
    
    return Nodes, maxnum
    

# Expand right child node
def Updaterigtchd(Nodes, pronum, maxnum, Xori, yori, clsnum, Xorit, yorit):
    # print("Running Updaterigtchd")
    """
    @brief Expand right child node.
    @param Nodes: Dictionary of nodes.
    @param pronum: Parent node index.
    @param maxnum: Maximum node index.
    @param Xori: Training features.
    @param yori: Training labels.
    @param clsnum: Number of classes.
    @param Xorit: Validation features.
    @param yorit: Validation labels.
    @return Updated Nodes and maxnum.
    """
    Rightidx = Nodes[pronum].falseidx
    Rightidxt = Nodes[pronum].falseidxt
    yright = yori[Rightidx]
    yrgini = Cptgininode(yright, clsnum)
    yrightt = yorit[Rightidxt]
    yrginit = Cptgininode(yrightt, clsnum)
    maxnum += 1
    Nodes[maxnum] = Node(maxnum)
    Nodes = Givetraintonode(Nodes, maxnum, Rightidx)
    Nodes = Givevaltonode(Nodes, maxnum, Rightidxt)
    yrcount = County(yright, clsnum)
    yrcountt = County(yrightt, clsnum)
    Nodes[maxnum].ycount = yrcount
    Nodes[maxnum].ycountt = yrcountt
    Nodes[maxnum].predcls = yrcountt.argmax()
    Nodes[maxnum].ginis = yrgini
    Nodes[maxnum].ginist = yrginit
    if yrginit == 0 or yrgini == 0:
        Nodes[maxnum].stoptrain = True
    else:
        Nodes[maxnum].stoptrain = False
    Nodes[pronum].rightchd = maxnum
    Nodes[maxnum].prntnb = pronum
    Nodes[maxnum].childtype = 'rightchild'
    
    return Nodes, maxnum


# Binary encoding of multi-class label
def Ecdlabel(yori, cnum):
    # print("Running Ecdlabel")
    """
    @brief Binary encoding of multi-class label.
    @param yori: Labels to encode.
    @param cnum: Classes present.
    @return Encoded labels.
    """
    # ynew = {}
    # for c in cnum:
    #     yc = np.zeros(yori.shape)
    #     yc[yori == c] = 1
    #     ynew[c] = yc
    # return ynew

    #Code for when yori is on GPU
    ynew = {}
    for c in cnum:
        c_int = int(c.item())  # Convert tensor to Python integer
        yc = torch.zeros_like(yori, dtype=torch.long, device=yori.device)  # Create a tensor of zeros on the same device
        yc[yori == c] = 1  # Use PyTorch-style boolean indexing
        ynew[c_int] = yc
    return ynew


# Gini index for classification at a node
def Cptginisplit(mds, X, y, T, clsnum):
    # print("Running Cptginisplit")
    """
    @brief Compute Gini index for classification at a node.
    @param mds: Model predictions.
    @param X: Data features.
    @param y: Labels.
    @param T: Number of time steps.
    @param clsnum: Number of classes.
    @return Gini index and split indices.
    """
    ginis = []
    for md in mds.values():
        Xmd_preds = md(X[:,:T], X[:,T:2*T], X[:,2*T:])
        Xmd_predsrd = torch.round(Xmd_preds)
        onesnum = torch.sum(Xmd_predsrd == 1.)
        ygroup1 = y[Xmd_predsrd == 1.]
        zerosnum = torch.sum(Xmd_predsrd == 0.)
        ygroup0 = y[Xmd_predsrd == 0.]
        ginimd = Cpt_ginigroup(onesnum, ygroup1, zerosnum, ygroup0, clsnum)
        ginis.append(ginimd)
    return ginis
   

# Gini index computation for each classifier
def Cpt_ginigroup(num1, y1, num0, y0, clsnum):
    # print("Running Cpt_ginigroup")
    """
    @brief Compute Gini index for each classifier group.
    @param num1: Number of samples in group 1.
    @param y1: Labels in group 1.
    @param num0: Number of samples in group 0.
    @param y0: Labels in group 0.
    @param clsnum: Number of classes.
    @return Gini index.
    """
    y1prob = torch.zeros(clsnum)
    y0prob = torch.zeros(clsnum)
    y1N = len(y1)
    y0N = len(y0)
    nums = num1 + num0
    for i in range(clsnum):
        if y1N>0:
            y1prob[i] = sum(y1==i)/y1N
        if y0N>0:
            y0prob[i] = sum(y0==i)/y0N

    ginipt1 = 1 - torch.sum(y1prob**2)
    ginipt0 = 1 - torch.sum(y0prob**2)
    ginirt = (num1/nums) * ginipt1 + (num0/nums) * ginipt0
    return ginirt


# Gini index for a node
def Cptgininode(yori, clsn):
    # print("Running Cptgininode")
    """
    @brief Compute Gini index for a node.
    @param yori: Labels at node.
    @param clsn: Number of classes.
    @return Gini index.
    """
    yfrac = np.zeros(clsn)
    for i in range(clsn):
        try:
            yfrac[i] = sum(yori==i)/len(yori)
        except:
            yfrac[i] = 0
    ginin = 1 - np.sum(yfrac ** 2)
    return ginin


# Accuracy for a node phase classifier
@torch.no_grad() #For everything inside this function, don't track gradients
def Cpt_Accuracy(mdl, X, y, T):
    """
    Compute binary accuracy entirely on device.
    Returns:
      preds: LongTensor of 0/1 (shape [N])
      acc:   Python float accuracy
      trueidx, falseidx: LongTensors of local indices
    """
    device = X.device
    logits = mdl(X[:, :T], X[:, T:2*T], X[:, 2*T:])
    preds = torch.round(logits).long().view(-1)
    trueidx = torch.where(preds == 1)[0]
    falseidx = torch.where(preds == 0)[0]
    acc = (preds == y.to(device)).float().mean().item()
    return preds, acc, trueidx, falseidx


# Count the number of data in each class
def County(yori, clsnum):
    # print("Running County")
    """
    @brief Count the number of data in each class.
    @param yori: Labels.
    @param clsnum: Number of classes.
    @return Array of counts per class.
    """
    # ycount = np.zeros((clsnum))
    # for i in range(clsnum):
    #     ycount[i] = sum(yori == i)
    # return ycount

    #Writing code for when yori is on GPU
    ycount = torch.zeros(clsnum, dtype=torch.long, device=yori.device)  # Create a tensor on the same device as yori
    for i in range(clsnum):
        ycount[i] = torch.sum(yori == i)  # Use torch.sum for element-wise comparison
    return ycount


# Prune a tree using validation data
def Prune_tree(Tree, Xval, yval):
    # print("Running Prune_tree")
    """
    @brief Prune a tree using validation data.
    @param Tree: Tree dictionary.
    @param Xval: Validation features.
    @param yval: Validation labels.
    @return Pruned tree.
    """
    Xpredclass, tstaccu, accuuptobst, keep_list = Postprune(Tree, Xval, yval)
    Tree_pruned = {}
    
    for node_idx in Tree.keys():
        if node_idx in keep_list:
            Tree_pruned[node_idx] = Tree[node_idx]
        else:
            if Tree[node_idx].prntnb in keep_list:
                Tree_pruned[node_idx] = Tree[node_idx]
                if hasattr(Tree_pruned[node_idx], "leftchd"):
                    del Tree_pruned[node_idx].leftchd 
                if hasattr(Tree_pruned[node_idx], "rightchd"):
                    del Tree_pruned[node_idx].rightchd
                
    return Tree_pruned


# Postprune nodes of a tree classifier
def Postprune(Nodes, Xtestori, ytestori):
    """
    @brief Postprune nodes of a tree classifier.
    @param Nodes: Tree dictionary.
    @param Xtestori: Test features (PyTorch tensor).
    @param ytestori: Test labels (PyTorch tensor).
    @return Pruned tree.
    """
    # Ensure Xtestori is a PyTorch tensor and move it to the correct device
    Xtestori = Xtestori.to(device)
    ytestori = ytestori.to(device)
    T = Xtestori.shape[1] // 3

    # Initialize variables
    Nodes[0].Testidx = torch.arange(len(ytestori), device=device, dtype=torch.long)
    Xpredclass = torch.zeros_like(ytestori, device=device)
    Xpredupto = torch.zeros_like(ytestori, device=device, dtype=torch.long)
    accuuptobst = 0
    keep_list, prune_list = [], []

    testnode = 0
    while testnode < len(Nodes):
        if hasattr(Nodes[testnode], 'bestmodel'):
            testidx = Nodes[testnode].Testidx
            Xtest = Xtestori[testidx, :]
            ytest = ytestori[testidx]

            # Get predictions and accuracy
            Xpred, accutest, trueidx, falseidx = Cpt_Accuracy(
                Nodes[testnode].bestmodel, Xtest, ytest, T
            )
            Xpredrd = torch.round(Xpred)

            # Update node attributes
            Nodes[testnode].testtrueidx = trueidx
            Nodes[testnode].testfalseidx = falseidx
            Nodes[testnode].Xpreds = Xpredrd
            Xpredupto[testidx] = Xpredrd

            # Calculate accuracy
            accuupto = (Xpredupto == ytestori).float().mean().item()
            if accuupto > accuuptobst:
                accuuptobst = accuupto
                keep_list.append(testnode)
            else:
                prune_list.append(testnode)

            Nodes[testnode].Xpredsupto = Xpredupto.clone()
            Nodes[testnode].testaccuupto = accuupto

            # Propagate indices to child nodes
            if hasattr(Nodes[testnode], 'leftchd'):
                Nodes[Nodes[testnode].leftchd].Testidx = testidx[trueidx]
            if hasattr(Nodes[testnode], 'rightchd'):
                Nodes[Nodes[testnode].rightchd].Testidx = testidx[falseidx]
        else:
            # Handle leaf nodes
            if not hasattr(Nodes[testnode], 'leftchd') and not hasattr(Nodes[testnode], 'rightchd'):
                Xpredclass[Nodes[testnode].Testidx] = Nodes[testnode].predcls.to(device)

        testnode += 1

    # Final accuracy check
    tstaccu = (Xpredclass == ytestori).float().mean().item()
    if tstaccu > accuuptobst:
        keep_list = list(Nodes.keys())

    return Xpredclass, tstaccu, accuuptobst, keep_list


# Evaluate model's performance using test data
def Evaluate_model(Nodes, Xtestori, ytestori):
    """Evaluate model's performance entirely on GPU without numpy conversions."""
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # ensure tensors on device
    # if not torch.is_tensor(Xtestori):
    #     Xtestori = torch.tensor(Xtestori, device=device)
    # else:
    #     Xtestori = Xtestori.to(device)
    # if not torch.is_tensor(ytestori):
    #     ytestori = torch.tensor(ytestori, device=device, dtype=torch.long)
    # else:
    #     ytestori = ytestori.to(device)
    clsnum = int(ytestori.max().item()) + 1
    T = Xtestori.shape[1] // 3

    # initialize indices and prediction accumulator
    Nodes[0].Testidx = torch.arange(len(ytestori), device=device, dtype=torch.long)
    Xpredupto = torch.zeros_like(ytestori, device=device, dtype=torch.long)

    # traverse nodes
    for node_idx in list(Nodes.keys()):
        node = Nodes[node_idx]
        if hasattr(node, 'bestmodel'):
            idxs = node.Testidx
            # tensorify indices
            if not torch.is_tensor(idxs):
                idxs = torch.tensor(idxs, device=device, dtype=torch.long)
            else:
                idxs = idxs.to(device)
            node.Testidx = idxs

            # select test batch
            Xtest = Xtestori[idxs]
            ytest = ytestori[idxs]

            # GPU-optimized prediction via Cpt_Accuracy
            preds, _, true_local, false_local = Cpt_Accuracy(node.bestmodel, Xtest, ytest, T)
            node.testtrueidx = idxs[true_local]
            node.testfalseidx = idxs[false_local]
            node.Xpreds = preds

            # accumulate true-class predictions
            Xpredupto[node.testtrueidx] = node.bstmdlclass
            # compute majority class for false-case
            counts = County(ytest, clsnum)
            maincls = torch.argmax(counts).item()
            Xpredupto[node.testfalseidx] = maincls

            # record intermediate results
            node.Xpredsupto = Xpredupto.clone()
            node.testaccuupto = (Xpredupto == ytestori).sum().item() / len(ytestori)

            # propagate indices to children
            if hasattr(node, 'leftchd'):
                Nodes[node.leftchd].Testidx = node.testtrueidx
            if hasattr(node, 'rightchd'):
                Nodes[node.rightchd].Testidx = node.testfalseidx

    # final accuracy
    tstaccu = (Xpredupto == ytestori).sum().item() / len(ytestori)
    return tstaccu
