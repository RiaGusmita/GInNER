#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import argparse
import torch

from data_loader import getData
from train import train, train_indobert
from model_testing import model_testing
import fasttext
import fasttext.util

from transformers import BertTokenizer, AutoModel


DEVICE = torch.device("cpu")
START_TAG = "<START>"
STOP_TAG = "<STOP>"

def main(mode, loss_function, hidden_layers, nheads, lr, dropout, regularization, weight_decay, n_epoch, save_every, word_emb_model, word_emb_dim, ner_model): 
    if regularization==True:
        weight_decay==weight_decay
    else:
        weight_decay==0
    tag_to_idx = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6, START_TAG: 7, STOP_TAG: 8}    
    
    print('Hyper paramters:')  
    print("Loss function: {}".format(loss_function))
    print("Learning rate: {}",format(lr))
    print("Dropout: {}",format(dropout))
    if regularization==True:
        print("Weight Decay: {}", format(weight_decay))
    print("n Epochs: {}", format(n_epoch))
    
    word_emb=""
    if word_emb_model=="fasttext":
        word_emb = fasttext.load_model('cc.id.300.bin')
        word_emb = fasttext.util.reduce_model(word_emb, word_emb_dim)
    elif word_emb_model=="indobert":
        tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
        model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
        word_emb = (tokenizer, model)
    
    if mode == "train" or mode =="test" or mode=="all":
        train_dataset, valid_dataset, test_dataset = getData()
        print('Data loading ...')
        if mode == "train":
            if word_emb_model=="indobert":
                train_indobert(train_dataset, valid_dataset, tag_to_idx, DEVICE, dropout, hidden_layers, nheads, n_epoch, lr, regularization, word_emb_model, model, tokenizer, word_emb_dim)
            else:
                train(train_dataset, valid_dataset, tag_to_idx, DEVICE, dropout, hidden_layers, nheads, n_epoch, lr, regularization, word_emb_model, word_emb, word_emb_dim)
        if mode == "test":
            #print(test_dataset)
            model_testing(test_dataset, tag_to_idx, DEVICE, dropout, hidden_layers, nheads, word_emb_model, word_emb, ner_model, word_emb_dim)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Indonesian NER')
    parser.add_argument("--mode", type=str, default="all", help="use which mode type: train/test/all")
    parser.add_argument("--loss_function", type=str, default="CE", help="use which loss type: CE (CrossEntropy)")
    parser.add_argument("--hidden_layers", type=int, default=2, help="the number of hidden layers")
    parser.add_argument("--nheads", type=int, default=3, help="the number of heads attention")
    parser.add_argument("--lr", type=float, default=0.005, help="use to define learning rate hyperparameter")
    parser.add_argument("--dropout", type=float, default='0.0', help="use to define dropout hyperparameter")
    parser.add_argument("--weight_decay", type=float, default='1e-5', help="use to define weight decay hyperparameter if the regularization set to True")
    parser.add_argument("--regularization", type=bool, default=False, help="use to define regularization: True/False")
    parser.add_argument("--save_every", type=int, default=1, help="save model in every n epochs")
    parser.add_argument("--n_epoch", type=int, default=50, help="train model in total n epochs")
    parser.add_argument("--word_emb_model", type=str, default="spacy", help="spacy/fasttext")
    parser.add_argument("--word_emb_dim", type=int, default=96, help="word embedding dimension")
    parser.add_argument("--ner_model", type=str, default="final_model.pt", help="word embedding dimension")
    
    
    args = parser.parse_args()
    main(args.mode, args.loss_function, args.hidden_layers, args.nheads, args.lr, args.dropout, args.regularization, args.weight_decay, \
         args.n_epoch, args.save_every, args.word_emb_model, args.word_emb_dim, args.ner_model)