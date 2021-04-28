#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from data_loader import getSentences, get_data_from_sentences, createFullSentence, create_graph_from_sentence_and_word_vectors, get_data_from_sentences_fasttext
from model import GInNER
import sys
from tqdm import tqdm
from torch import optim
import torch
import os
import os.path as path
import matplotlib.pyplot as plt

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

def train(train_dataset, validation_dataset, device, dropout, hidden_layer, nheads, epochs, lr, regularization, word_emb_model, word_emb, word_embedding_dim,
          saving_dir="models", weight_decay=1e-5):
    
    if not path.exists(saving_dir):
        os.makedirs(saving_dir)
    
    sentences = getSentences(train_dataset)
    val_sentences = getSentences(validation_dataset)
    
    if word_emb_model =="fasttext":
        data = get_data_from_sentences_fasttext(sentences, word_emb)
        val_data = get_data_from_sentences_fasttext(val_sentences, word_emb)
    else:
        data = get_data_from_sentences(sentences)
        val_data = get_data_from_sentences(val_sentences)
    
    loss_function = torch.nn.CrossEntropyLoss()
    
    print("total data", len(data))
    #buckets = bin_data_into_buckets(data, bucket_size)
    #print("buckets", len(buckets))
    ginner = GInNER(word_embedding_dim, device, dropout, hidden_layer, nheads)
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
        #with torch.no_grad():
        for item in tqdm(data):
            try:
                optimizer.zero_grad()    
                words = item[0]
                labels = torch.LongTensor(item[2])
                word_embeddings = item[1]
                sentence = createFullSentence(words)
                A, X = create_graph_from_sentence_and_word_vectors(sentence, word_embeddings)
                output_tensor = ginner(X, A)
                loss = loss_function(output_tensor, labels).to(device)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_sentences +=1
                
            except:
                pass
        total_loss = total_loss/total_sentences
        
        ginner.eval()
        total_val_sentences = 0
        total_val_loss = 0
        total_val_acc = 0
        #with torch.no_grad():
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
                    total_val_loss += val_loss.item()
                        
                    acc_val = accuracy(output_tensor, labels)
                    total_val_acc += acc_val
                    total_val_sentences +=1
                except:
                    pass
        total_val_loss = total_val_loss/total_val_sentences
        total_val_acc = total_val_acc/total_val_sentences
        torch.save({
                    "epoch": i,
                    "model_state_dict": ginner.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss
                    }, path.join(saving_dir, "checkpoint_epoch_{}.pt".format(i)))
        print("epoch: {}".format(i), "training loss", total_loss, "validation loss", total_val_loss, "acc", total_val_acc)
        losses['train set'].append(total_loss)
        losses['val set'].append(total_val_loss)
        showPlot(arEpochs, losses, "training_val_loss")
        
def accuracy(output, labels):
    #logits_scores, pred = torch.max(output, 1, keepdim=True)
    #logits_label = logits_tags.detach().cpu().numpy().tolist()
    #y_pred = [predict[0] for predict in logits_label]
    
    preds = output.max(1)[1].type_as(labels)
    #print(preds)
    #print(labels)
    
    correct = preds.eq(labels).double()
    #print(correct)
    correct = correct.sum()
    #print(correct)
    correct = correct.detach().cpu().numpy()
    acc = correct / len(labels)
    #print(acc)
    #print("###")
    return acc 

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