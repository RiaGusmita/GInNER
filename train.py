#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from data_loader import getSentences, get_data_from_sentences, createFullSentence, create_graph_from_sentence_and_word_vectors, get_data_from_sentences_fasttext, get_data_from_sentences_indobert
from model import GInNER
import sys
from tqdm import tqdm
from torch import optim
import torch
import os
import os.path as path
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np
import random

def get_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def bin_data_into_buckets(data, batch_size):
    buckets = []
    size_to_data_dict = {}
    for item in data:
        sequence = item[0]
        length = len(sequence)
        try:
            size_to_data_dict[length].append(item)
        except:
            size_to_data_dict[length] = [item]
    for key in size_to_data_dict.keys():
        data = size_to_data_dict[key]
        chunks = get_chunks(data, batch_size)
        for chunk in chunks:
            buckets.append(chunk)
    return buckets

def onehot(labels: torch.Tensor, label_num):
    return torch.zeros(labels.shape[0], label_num, device=labels.device).scatter_(1, labels.view(-1, 1), 1)

def loss_fn(preds:list, labels):
    start_token_labels, end_token_labels = labels.split(1, dim = -1)
    start_token_labels = start_token_labels.squeeze(-1)
    end_token_labels = end_token_labels.squeeze(-1)

    print('*'*50)
    print(preds[0].shape) # preds [0] and [1] has the same shape and dtype
    print(preds[0].dtype) # preds [0] and [1] has the same shape and dtype
    print(start_token_labels.shape) # labels [0] and [1] has the same shape and dtype
    print(start_token_labels.dtype) # labels [0] and [1] has the same shape and dtype

    start_loss = torch.nn.CrossEntropyLoss()(preds[0], start_token_labels)
    end_loss = torch.nn.CrossEntropyLoss()(preds[1], end_token_labels)

    avg_loss = (start_loss + end_loss) / 2
    return avg_loss

def train(train_dataset, validation_dataset, tag_to_idx, device, dropout, hidden_layer, nheads, epochs, lr, regularization, word_emb_model, word_emb, word_embedding_dim,
          saving_dir="models", weight_decay=1e-5):
    
    if not path.exists(saving_dir):
        os.makedirs(saving_dir)
    
    print_to = 'logs.txt'
    with open(print_to, 'w+') as f:
        f.write("Starting Training \n")
    
    sentences = getSentences(train_dataset)
    val_sentences = getSentences(validation_dataset)
    
    if word_emb_model =="fasttext" or word_emb_model=="fasttext-indobert":
        data = get_data_from_sentences_fasttext(sentences, word_emb, tag_to_idx)
        val_data = get_data_from_sentences_fasttext(val_sentences, word_emb, tag_to_idx)
    elif word_emb_model =="indobert":
        tokenizer, model = word_emb
        data = get_data_from_sentences_indobert(sentences, tokenizer, model, tag_to_idx)
        val_data = get_data_from_sentences_indobert(val_sentences, tokenizer, model, tag_to_idx)
    else:
        data = get_data_from_sentences(sentences, tag_to_idx)
        val_data = get_data_from_sentences(val_sentences, tag_to_idx)
    
    loss_function = torch.nn.CrossEntropyLoss()
    
    print("total data", len(data))
    #buckets = bin_data_into_buckets(data, bucket_size)
    #print("buckets", len(buckets))
    ginner = GInNER(word_embedding_dim, tag_to_idx, device, dropout, hidden_layer, nheads)
    print(ginner)
    if regularization:
        optimizer = optim.Adam(ginner.parameters(), lr=lr, weight_decay=weight_decay)
    else:    
        optimizer = optim.Adam(ginner.parameters(), lr=lr)
    
    arEpochs = []
    losses = {'Training set':[], 'Validation set': []}
    
    best_f1_score_micro = 0
    for i in range(1,epochs+1):
        arEpochs.append(i)
        sys.stderr.write('--------- Epoch ' + str(i) + ' ---------\n')
        ginner.train()
        total_sentences = 0
        total_loss = 0
        broken_sentence = 0
        for item in tqdm(data):
            try:
                words = item[0]
                labels = torch.LongTensor(item[2])
                word_embeddings = item[1]
                sentence = createFullSentence(words)
                A, X = create_graph_from_sentence_and_word_vectors(sentence, word_embeddings)
                #logit_scores, logit_tags = ginner(X, A)
                output_tensor = ginner(X, A)
                loss = loss_function(output_tensor, labels).to(device)
                #print("tags", len(logit_tags))
                #print("labels", len(labels.detach().cpu().numpy()))
                #loss = loss_function(output_tensor, labels).to(device)
                #loss = ginner.neg_log_likelihood(X, A, labels)    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                total_loss += loss.item()
                total_sentences +=1
            except:
                broken_sentence += 1
                pass
        total_loss = total_loss/total_sentences
        print("broken sentence during training", broken_sentence)
        ginner.eval()
        total_val_sentences = 0
        total_val_loss = 0
        total_val_acc = 0
        #with torch.no_grad():
        list_f1_score_micro = []
        
        broken_sentence = 0
        with torch.no_grad():
            for item in tqdm(val_data):
                try:
                    words = item[0]
                    labels = torch.LongTensor(item[2])
                    word_embeddings = item[1]
                    sentence = createFullSentence(words)
                    A, X = create_graph_from_sentence_and_word_vectors(sentence, word_embeddings)
                        #logits_scores, logits_tags = ginner(X, A)
                    output_tensor = ginner(X, A)
                    val_loss = loss_function(output_tensor, labels).to(device)
                        #val_loss = ginner.neg_log_likelihood(X, A, labels) 
                        #logits_scores, logits_tags = torch.max(output_tensor, 1, keepdim=True)
                        #logits_tags = logits_tags.detach().cpu().numpy().tolist()
                    y_pred = output_tensor.max(1)[1].type_as(labels)
                        #print("y_pred", y_pred, len(y_pred))
                    y_true = labels.detach().cpu().numpy()
                        #print("y_true", y_true, len(y_true))
                    f1_score_micro = f1_score(y_true, y_pred, average='micro')
                        #print("f1 score micro", f1_score_micro)
                        #score = logit_scores.detach().cpu().numpy()
                    list_f1_score_micro.append(f1_score_micro)
                                
                    total_val_loss += val_loss.item()
                    total_val_acc += accuracy_alt(y_pred, labels)
                    total_val_sentences +=1
                except:
                    broken_sentence +=1
                    pass
        total_val_loss = total_val_loss/total_val_sentences
        total_val_acc = total_val_acc/total_val_sentences
        avg_f1_scores_micro = sum(list_f1_score_micro)/len(list_f1_score_micro)
        if avg_f1_scores_micro > best_f1_score_micro:
            best_f1_score_micro = avg_f1_scores_micro
            torch.save({
                    "epoch": i,
                    "model_state_dict": ginner.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss
                    }, path.join(saving_dir, "best_model.pt".format(i)))
            evaluate_randomly(val_data, ginner, tag_to_idx)
    
        torch.save({
                    "epoch": i,
                    "model_state_dict": ginner.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss
                    }, path.join(saving_dir, "checkpoint_epoch_{}.pt".format(i)))
        torch.save({
                    "epoch": i,
                    "model_state_dict": ginner.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss
                    }, path.join(saving_dir, "final_model.pt".format(i)))
        print("broken sentence during testing", broken_sentence)
        print("epoch: {}".format(i), "training loss", total_loss, "validation loss", total_val_loss, "acc", total_val_acc, "f1-micro", avg_f1_scores_micro)
        with open(print_to, 'a') as f:
            f.write("epoch {} train_loss {} validation_loss {} acc {} f1-micro {} \n".format(i, total_loss, total_val_loss, total_val_acc, avg_f1_scores_micro))
        losses['Training set'].append(total_loss)
        losses['Validation set'].append(total_val_loss)
        showPlot(arEpochs, losses, "training_val_loss")
        
def accuracy_alt(outputs, labels):
    correct = 0
    labels = labels.detach().cpu().numpy()
    #print("outputs", outputs)
    #print("labels", labels)
    #print("len output", len(outputs))
    #print("len labels", len(labels))
    for i, output in enumerate(outputs):
        if output == labels[i]:
            correct += 1
    acc = correct / len(labels)
    return acc 

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    labels = labels.detach().cpu().numpy()
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
'''Used to plot the progress of training. Plots the loss value vs. time'''
def showPlot(epochs, losses, fig_name):
    colors = ('red','blue')
    x_axis_label = 'Epochs'
    i = 0
    for key, losses in losses.items():
      if len(losses) > 0:
        plt.plot(epochs, losses, label=key, color=colors[i])
        i += 1
    plt.legend(loc='upper left')
    plt.xlabel(x_axis_label)
    plt.ylabel('Loss')
    plt.title('Training Results')
    plt.savefig(fig_name+'.png')
    plt.close('all')

def train_indobert(train_dataset, validation_dataset, tag_to_idx, device, dropout, hidden_layer, nheads, epochs, lr, regularization, word_emb_model, model, tokenizer, word_embedding_dim,
          saving_dir="models", weight_decay=1e-5):
    
    if not path.exists(saving_dir):
        os.makedirs(saving_dir)
    
    sentences = getSentences(train_dataset)
    val_sentences = getSentences(validation_dataset)
    data = get_data_from_sentences_indobert(sentences, tokenizer, model, tag_to_idx)
    val_data = get_data_from_sentences_indobert(val_sentences, tokenizer, model, tag_to_idx)
    
    loss_function = torch.nn.CrossEntropyLoss()
    
    print("total data", len(data))
    word_embedding_dim = 768
    ginner = GInNER(word_embedding_dim, tag_to_idx, device, dropout, hidden_layer, nheads)
    print(ginner)
    if regularization:
        optimizer = optim.Adam(ginner.parameters(), lr=lr, weight_decay=weight_decay)
    else:    
        optimizer = optim.Adam(ginner.parameters(), lr=lr)
    
    arEpochs = []
    losses = {'train set':[], 'val set': []}
    for i in range(1,epochs+1):
        arEpochs.append(i)
        sys.stderr.write('--------- Epoch ' + str(i) + ' ---------\n')
        ginner.train()
        total_sentences = 0
        total_loss = 0
        broken_sentence = 0
        for item in tqdm(data):
            try:   
                words = item[0]
                labels = torch.LongTensor(item[2])
                word_embeddings = item[1]
                sentence = createFullSentence(words)
                A, X = create_graph_from_sentence_and_word_vectors(sentence, word_embeddings)
                output_tensor = ginner(X, A)
                loss = loss_function(output_tensor, labels).to(device)
                        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                total_loss += loss.item()
                total_sentences +=1   
            except:
                broken_sentence +=1
                pass
        total_loss = total_loss/total_sentences
        print("broken_sentence during training", broken_sentence)
        ginner.eval()
        total_val_sentences = 0
        total_val_loss = 0
        total_val_acc = 0
        #with torch.no_grad():
        list_f1_score_micro = []
        best_f1_score_micro = 0
        broken_sentence = 0
        with torch.no_grad():
            for item in tqdm(val_data):
                try:
                    words = item[0]
                    labels = torch.LongTensor(item[2])
                    word_embeddings = item[1]
                    sentence = createFullSentence(words)
                    A, X = create_graph_from_sentence_and_word_vectors(sentence, word_embeddings)
                    output_tensor = ginner(X, A)
                    val_loss = loss_function(output_tensor, labels).to(device)
                    
                    logits_scores, logits_tags = torch.max(output_tensor, 1, keepdim=True)
                    logits_label = logits_tags.detach().cpu().numpy().tolist()
                    y_pred = [predict[0] for predict in logits_label]
                    y_true = labels.detach().cpu().numpy().tolist()
                    f1_score_micro = f1_score(y_true, y_pred, average='micro')
                    list_f1_score_micro.append(f1_score_micro)
                    
                    total_val_loss += val_loss.item()
                    acc_val = accuracy(output_tensor, labels)
                    total_val_acc += acc_val
                    total_val_sentences +=1
                except:
                    broken_sentence +=1
                    pass
        total_val_loss = total_val_loss/total_val_sentences
        total_val_acc = total_val_acc/total_val_sentences
        avg_f1_scores_micro = sum(list_f1_score_micro)/len(list_f1_score_micro)
        if avg_f1_scores_micro > best_f1_score_micro:
            torch.save({
                    "epoch": i,
                    "model_state_dict": ginner.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss
                    }, path.join(saving_dir, "best_model.pt".format(i)))
            
    
        torch.save({
                    "epoch": i,
                    "model_state_dict": ginner.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss
                    }, path.join(saving_dir, "checkpoint_epoch_{}.pt".format(i)))
        torch.save({
                    "epoch": i,
                    "model_state_dict": ginner.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss
                    }, path.join(saving_dir, "final_model.pt".format(i)))
        print("broken sentence during validation", broken_sentence)
        print("epoch: {}".format(i), "training loss", total_loss, "validation loss", total_val_loss, "acc", total_val_acc)
        losses['train set'].append(total_loss)
        losses['val set'].append(total_val_loss)
        showPlot(arEpochs, losses, "training_val_loss")
    
def word_subword_tokenize(sentence, tokenizer):
    # Add CLS token
    subwords = []#[tokenizer.cls_token_id]
    subword_to_word_indices = []#[-1] # For CLS

    # Add subwords
    for word_idx, word in enumerate(sentence.split(" ")):
        subword_list = tokenizer.encode(word, add_special_tokens=False)
        if len(subword_list)>1:
            subword_list = [round(np.average(subword_list))]
        subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            
        subwords += subword_list

    return subwords, subword_to_word_indices

def evaluate_randomly(data, model, tag_to_idx):
    try:
        item = random.choice(data)
        
        words = item[0]
        word_embeddings = item[1]
        sentence = createFullSentence(words)
        labels = item[2]
        A, X = create_graph_from_sentence_and_word_vectors(sentence, word_embeddings)
        output_tensor = model(X, A)
        #output_tensor = output_tensor
        logits_scores, logit_tags = torch.max(output_tensor, 1, keepdim=True)
        #logit_scores, logit_tags = model(X, A)
        #logits_label = logits_tags.detach().cpu().numpy().tolist()
        y_pred = [predict for predict in logit_tags]
        key_list = list(tag_to_idx.keys())
        value_list = list(tag_to_idx.values())
        for i, word in enumerate(words):
            tag = value_list.index(y_pred[i])
            label_tag = value_list.index(labels[i])
            print("word {} prediction {} True label {}".format(word, key_list[tag], key_list[label_tag]))
    except:
        pass