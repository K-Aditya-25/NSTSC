# -*- coding: utf-8 -*-
"""
@file NSTSC_main.py
@brief Main script to train and evaluate the NSTSC model on a dataset.
"""

from Models_node import *
from utils.datautils import *
from utils.train_utils import *
import time
import sys
import os
import pickle
import torch

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    """
    @brief Main function to train and evaluate the NSTSC model.
    """
    Dataset_name = sys.argv[1] if len(sys.argv) > 1 else "Coffee"
    print('Start Training ---' + str(Dataset_name) + ' ---dataset\n')
    dataset_path_ = "../UCRArchive_2018/"
    normalize_dataset = True
    Max_epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"Epochs = {Max_epoch}")
    # Data Preprocessing and preparation on CPU
    preprocess_start = time.time() 
    Xtrain_raw, ytrain_raw, Xval_raw, yval_raw, Xtest_raw, ytest_raw = Readdataset(dataset_path_, Dataset_name)
    Xtrain, Xval, Xtest = Multi_view(Xtrain_raw, Xval_raw, Xtest_raw)
    N, T = calculate_dataset_metrics(Xtrain)
    #Convert nd arrays to tensors and shift to MPS
    # Xtrain = torch.tensor(Xtrain, device=device, dtype=torch.float16)
    # Xval = torch.tensor(Xval, device=device, dtype=torch.float16)
    # Xtest = torch.tensor(Xtest, device=device, dtype=torch.float16)
    # ytrain_raw = torch.tensor(ytrain_raw, device=device, dtype=torch.int64)
    # yval_raw = torch.tensor(yval_raw, device=device, dtype=torch.int64)
    # ytest_raw = torch.tensor(ytest_raw, device=device, dtype=torch.int64)
    preprocess_end = time.time()
    print("Time took for preprocessing data: {}s".format(preprocess_end - preprocess_start))
    #Model Training
    train_start = time.time()
    Tree = Train_model(Xtrain, Xval, ytrain_raw, yval_raw, epochs=Max_epoch, normalize_timeseries=normalize_dataset)
    train_end = time.time()
    print("Time took for training model: {}s".format(train_end - train_start))
    #Save the learned model weights as pkl file
    os.makedirs("../Tree_Models", exist_ok=True)
    with open(f"../Tree_Models/{Dataset_name}_tree.pkl", "wb") as f:
        pickle.dump(Tree, f)
    # model testing/evaluation
    eval_start = time.time()
    testaccu = Evaluate_model(Tree, Xtest, ytest_raw)
    eval_end = time.time()
    print("Test accuracy for dataset {} is --- {}".format(Dataset_name, testaccu))
    print("Time took for testing model: {}s".format(eval_end - eval_start))

if __name__ == "__main__":
    main()

