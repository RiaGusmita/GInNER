#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from model import GInNER
import os.path as path
import torch
from data_loader import getSentences, get_data_from_sentences, createFullSentence, create_graph_from_sentence_and_word_vectors, get_class_name, get_data_from_sentences_fasttext, get_data_from_sentences_indobert
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score
import logging
import random

_logger = logging.getLogger(__name__)

def model_testing(test_dataset, tag_to_idx, device, dropout, hidden_layer, nheads, word_emb_model, word_emb, ner_model, word_embedding_dim=96):
    #print(test_dataset)
    
    sentences = getSentences(test_dataset)
    #print("sentences", sentences)
    #print('len sentences', len(sentences))
    if word_emb_model =="fasttext":
        data = get_data_from_sentences_fasttext(sentences, word_emb, tag_to_idx)
    elif word_emb_model =="indobert":
        tokenizer, model = word_emb
        data = get_data_from_sentences_indobert(sentences, tokenizer, model, tag_to_idx)
    else:
        data = get_data_from_sentences(sentences, tag_to_idx)
    ginner = GInNER(word_embedding_dim, tag_to_idx, device, dropout, hidden_layer, nheads)
    checkpoint = torch.load(path.join("models", ner_model))
    ginner.load_state_dict(checkpoint["model_state_dict"])
    print(ginner)
    ginner.to(device)
    #print('len data', len(data))
    recall_scores = []
    precision_scores = []
    f1_scores = []
    broken_sentences=0
    for item in tqdm(data):
        words = item[0]
        labels = torch.LongTensor(item[2])
        word_embeddings = item[1]
        sentence = createFullSentence(words)
        try:           
            A, X = create_graph_from_sentence_and_word_vectors(sentence, word_embeddings)
            #output_tensor = ginner(X, A)
            #output_tensor = output_tensor
            logit_scores, logit_tags = ginner(X, A)
            #logits_scores, logits_tags = torch.max(output_tensor, 1, keepdim=True)
            #logits_label = logits_tags.detach().cpu().numpy().tolist()
            y_pred = [predict for predict in logit_tags]
            y_true = labels.detach().cpu().numpy()
            print("y_pred", y_pred)
            print("y_true", y_true)
            recall_scores.append(recall_score(y_true, y_pred, average='macro'))
            precision_scores.append(precision_score(y_true, y_pred, average='macro'))
            f1_scores.append(f1_score(y_true, y_pred, average='macro'))
        except:
            _logger.warning('Cannot process the following sentence: ' + sentence)
            broken_sentences += 1
            continue
        
    avg_recall = sum(recall_scores)/len(recall_scores)
    avg_precision = sum(precision_scores)/len(precision_scores)
    avg_fscores = sum(f1_scores)/len(f1_scores)
    print("total sentences", len(sentences))
    print("Broken sentences", broken_sentences)
    print("precision {}, recall {}, f1 {}".format(avg_precision, avg_recall, avg_fscores))
    evaluate_randomly(data, ginner, tag_to_idx)
    
def evaluate_randomly(data, model, tag_to_idx):
    item = random.choice(data)
    
    words = item[0]
    word_embeddings = item[1]
    sentence = createFullSentence(words)
    labels = item[2]
    A, X = create_graph_from_sentence_and_word_vectors(sentence, word_embeddings)
    #output_tensor = model(X, A)
    #output_tensor = output_tensor
    #logits_scores, logits_tags = torch.max(output_tensor, 1, keepdim=True)
    logit_scores, logit_tags = model(X, A)
    #logits_label = logits_tags.detach().cpu().numpy().tolist()
    y_pred = [predict for predict in logit_tags]
    for i, word in enumerate(words):
        print("word {} prediction {} True label {}".format(word, tag_to_idx[y_pred[i]], tag_to_idx[labels[i]]))

def test_ner(sentence):
    
    words = item[0]
    word_embeddings = item[1]
    sentence = createFullSentence(words)
    labels = item[2]
    A, X = create_graph_from_sentence_and_word_vectors(sentence, word_embeddings)
    output_tensor = model(X, A)
    output_tensor = output_tensor
    logits_scores, logits_tags = torch.max(output_tensor, 1, keepdim=True)
    logits_label = logits_tags.detach().cpu().numpy().tolist()
    y_pred = [predict[0] for predict in logits_label]
    for i, word in enumerate(words):
        print("word {} prediction {} True label {}".format(word, get_class_name(y_pred[i]), get_class_name(labels[i])))