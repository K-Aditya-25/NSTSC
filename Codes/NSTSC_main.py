# -*- coding: utf-8 -*-
"""
@file NSTSC_main.py
@brief Main script to train and evaluate the NSTSC model on a dataset.
"""

from Models_node import *
from utils.datautils import *
from utils.train_utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    """
    @brief Main function to train and evaluate the NSTSC model.
    """
    Dataset_name = "Coffee"
    print('Start Training ---' + str(Dataset_name) + ' ---dataset\n')
    dataset_path_ = "../UCRArchive_2018/"
    normalize_dataset = True
    Max_epoch = 2
    # model training
    Xtrain_raw, ytrain_raw, Xval_raw, yval_raw, Xtest_raw, ytest_raw = Readdataset(dataset_path_, Dataset_name)
    Xtrain, Xval, Xtest = Multi_view(Xtrain_raw, Xval_raw, Xtest_raw)

    # Convert NumPy arrays to PyTorch tensors and move to GPU
    Xtrain_gpu = torch.tensor(Xtrain, dtype=torch.float32).to(device)
    Xval_gpu = torch.tensor(Xval, dtype=torch.float32).to(device)
    Xtest_gpu = torch.tensor(Xtest, dtype=torch.float32).to(device)
    # Convert labels to PyTorch tensors and move to GPU
    ytrain_raw_gpu = torch.tensor(ytrain_raw, dtype=torch.long).to(device)
    yval_raw_gpu = torch.tensor(yval_raw, dtype=torch.long).to(device)
    ytest_raw_gpu = torch.tensor(ytest_raw, dtype=torch.long).to(device)

    N, T = calculate_dataset_metrics(Xtrain)
    # Tree = Train_model(Xtrain, Xval, ytrain_raw, yval_raw, epochs=Max_epoch, normalize_timeseries=normalize_dataset)
    Tree = Train_model(Xtrain_gpu, Xval_gpu, ytrain_raw_gpu, yval_raw_gpu, epochs=Max_epoch, normalize_timeseries=normalize_dataset)
    # model testing
    # testaccu = Evaluate_model(Tree, Xtest, ytest_raw)
    testaccu = Evaluate_model(Tree, Xtest_gpu, ytest_raw_gpu)
    print("Test accuracy for dataset {} is --- {}".format(Dataset_name, testaccu))


if __name__ == "__main__":
    main()

