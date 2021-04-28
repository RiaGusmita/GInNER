#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from tqdm import tqdm
import numpy as np
import spacy
import torch
#print(parser)

import fasttext
import fasttext.util

#ft = fasttext.load_model('cc.id.300.bin')

parser = spacy.load("id_spacy")


classes = ["O", "B-PERSON", "I-PERSON", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
_partial_word = '%pw'


def readData(filename):
    data = []
    with open(filename, 'r') as f:
        for line in tqdm(f):
            values = line.split("\t")
            #print(values[0], len(values[0].strip()))
            if len(values[0].strip())>=1:
                word = values[0]
                label = values[1]
                label = re.sub("\n", "", label)
                label.strip()
                items = [word, label]
            else:
                items = ["<EOS>", "<EOS>"]
            data.append(items)
    return data

def getData():
    train_filename = "datasets/train.bio"
    dev_filename = "datasets/dev.bio"
    test_filename = "datasets/test.bio"
    train = readData(train_filename)
    dev = readData(dev_filename)
    test = readData(test_filename)
    #print(test)
    
    return train, dev, test

def createFullSentence(words):
    sentence = ' '.join(words)
    sentence = re.sub(r' (\'[a-zA-Z])', r'\1', sentence)
    sentence = re.sub(r' \'([0-9])', r' \1', sentence)
    sentence = re.sub(r' (,.)', r'\1', sentence)
    sentence = re.sub(r' " (.*) " ', r' "\1" ', sentence)
   
    return sentence

def getSentences(data):
    words_and_label = []
    sentences = []
    for item in data:
        #print(item)
        if item[0] != "<EOS>":
            word = item[0]
            label = item[1]
            words_and_label.append((word, label))
        else:
            sentences.append(words_and_label)
            words_and_label = []
    return sentences

def get_data_from_sentences(sentences, word_emb_model):
    all_data = []
    A = np.zeros((len(classes) + 1, len(classes) + 1))
    total_tokens = 0
    for sentence in tqdm(sentences):
        word_data = []
        class_data = []
        class_text = []
        words = []
        prior_entity = len(classes)
        #print('sentence', sentence)
        for word, entity in sentence:
            if word == _partial_word:
                continue
            words.append(word)
            word_vector = get_word_vector(word, word_emb_model)
            vector = word_vector
            entity_num = get_entity_num(entity)
            word_data.append(vector)
            class_data.append(entity_num)
            class_text.append(entity)
            A[prior_entity, entity_num] += 1
            prior_entity = entity_num
            total_tokens += 1
        all_data.append((words, word_data, class_data, class_text))
    return all_data

def get_word_vector(word, word_emb_model):
    if word_emb_model =="fasttext":
        parser = ft
        try:
            vector = parser.get_word_vector(word)
        except:
            vector = np.zeros([parser.get_dimension(),])
    
    if word_emb_model=="spacy":
        parsed = parser(word)
        default_vector = parser('entity')[0].vector
        try:
            vector = parsed[0].vector
            #print(vector.shape)
            if vector_is_empty(vector):
                vector = default_vector
        except:
            vector = default_vector
        
    return np.array(vector, dtype=np.float64) 

def vector_is_empty(vector):
    to_throw = 0
    for item in vector:
        if item == 0.0:
            to_throw += 1
    if to_throw == len(vector):
        return True
    return False  

def get_class_vector(class_name):
    vector = [0.] * (len(classes) + 1)
    index = len(classes)
    try:
        index = classes.index(class_name)
    except:
        pass
    vector[index] = 1.
    return vector


def get_entity_num(class_name):
    entity_num = len(classes)
    try:
        entity_num = classes.index(class_name)
    except:
        pass
    return entity_num

def get_class_name(class_num):
    #print(classes)
    entity = classes[class_num]
    return entity

def create_graph_from_sentence_and_word_vectors(sentence, word_vectors):
    if not isinstance(sentence, str):
        raise TypeError("String must be an argument")
    from igraph import Graph
    from nl import SpacyTagger as Tagger, SpacyParser as Parser
    from scipy.sparse import coo_matrix

    sentence = re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r' \1_\2 ', sentence)
    
    tagger = Tagger(sentence)
    parser = Parser(tagger)

    X = []
    for i in range(len(word_vectors)):
        X.append(word_vectors[i])
    X = np.array(X)

    nodes, edges, words, tags, types = parser.execute()
    g = Graph(directed=True)
    g.add_vertices(nodes)
    g.add_edges(edges)
    A = np.array(g.get_adjacency().data)
    A = coo_matrix(A)

    #print(X)
    return A, torch.tensor(X)

def get_words_embeddings_from_sentence(sentence, word_emb_model):
    sentence = re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r' \1_\2 ', sentence)
    if word_emb_model == "spacy":
        tokens = parser(sentence)
        print(tokens)
    
    if word_emb_model == "fasttext":
        parser = ft
        
    return _get_word_vectors_from_tokens(tokens)


def _get_word_vectors_from_tokens(tokens):
    words = []
    vectors = []
    idx = []
    for token in tokens:
        word = token.orth_
        #tag = token.tag_
        words.append(word)
        vectors.append(get_word_vector(word))
        idx.append([token.idx, token.idx + len(token.orth_)])
    return words, vectors, idx