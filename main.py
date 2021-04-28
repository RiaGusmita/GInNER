#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import argparse
import torch

from data_loader import getData
from train import train
from model_testing import model_testing


DEVICE = torch.device("cpu")

def main(mode, loss_function, hidden_layers, nheads, lr, dropout, regularization, weight_decay, n_epoch, save_every): 
    if regularization==True:
        weight_decay==weight_decay
    else:
        weight_decay==0
        
    print('Hyper paramters:')  
    print("Loss function: {}".format(loss_function))
    print("Learning rate: {}",format(lr))
    print("Dropout: {}",format(dropout))
    if regularization==True:
        print("Weight Decay: {}", format(weight_decay))
    print("n Epochs: {}", format(n_epoch))
    
    if mode == "train" or mode =="test" or mode=="all":
        train_dataset, valid_dataset, test_dataset = getData()
        print('Data loading ...')
        if mode == "train":
            train(train_dataset, valid_dataset, DEVICE, dropout, hidden_layers, nheads, n_epoch, lr, regularization)
        if mode == "test":
            #print(test_dataset)
            model_testing(test_dataset, DEVICE, dropout, hidden_layers, nheads)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Indonesian NER')
    parser.add_argument("--mode", type=str, default="all", help="use which mode type: train/test/all")
    parser.add_argument("--loss_function", type=str, default="BCE", help="use which loss type: BCE/MSE")
    parser.add_argument("--hidden_layers", type=int, default=2, help="the number of hidden layers")
    parser.add_argument("--nheads", type=int, default=3, help="the number of heads attention")
    parser.add_argument("--lr", type=float, default=0.005, help="use to define learning rate hyperparameter")
    parser.add_argument("--dropout", type=float, default='0.0', help="use to define dropout hyperparameter")
    parser.add_argument("--weight_decay", type=float, default='1e-5', help="use to define weight decay hyperparameter if the regularization set to True")
    parser.add_argument("--regularization", type=bool, default=False, help="use to define regularization: True/False")
    parser.add_argument("--save_every", type=int, default=1, help="save model in every n epochs")
    parser.add_argument("--n_epoch", type=int, default=50, help="train model in total n epochs")
    
    args = parser.parse_args()
    main(args.mode, args.loss_function, args.hidden_layers, args.nheads, args.lr, args.dropout, args.regularization, args.weight_decay, \
         args.n_epoch, args.save_every)