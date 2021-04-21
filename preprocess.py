#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 10:52:51 2021

@author: asep
"""

import re
import numpy as np
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from nlp_id.tokenizer import Tokenizer 
from nlp_id.postag import PosTag
import sys

from gensim.models.keyedvectors import KeyedVectors

from gcn_ner.ner_model import GCNNerModel
import gcn_ner.utils.aux as aux

_partial_word = '%pw'

def get_data(filename):
    items = []
    sentences = []
    with open(filename, 'r') as f:
        for line in f:
            values = line.split("\t")
            #print(values)
            #print(len(values))
            if len(values) >2:
                word = values[2]
                ne = values[9]
                tag = values[3]
                ne = re.sub("\n", "", ne)
                if ne =="_":
                    ne="O"
                
                items.append((word, tag, ne))
            else:
                if len(items)>0:
                    sentences.append(items)
                items=[]
    return sentences

def build_dict(sentences):
    words_dict = {}
    ne_dict = {}
    tags_dict={}
    #print(words)
    for sentence in sentences:
        for word, tag, ne in sentence:
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            if ne not in ne_dict:
                ne_dict[ne] = len(ne_dict)
            if tag not in tags_dict:
                tags_dict[tag] = len(tags_dict)    
    return words_dict, tags_dict, ne_dict

def get_class_vector(class_name, ne_dict):
    vector = [0.] * (len(ne_dict) + 1)
    index = len(ne_dict)
    try:
        index = ne_dict[class_name]
        #print(index)
    except:
        pass
    vector[index] = 1.
    return vector
def get_tagging_vector(tag, tags):
    vector = [0.] * (len(tags) + 1)
    index = len(tags)
    try:
        index = tags[tag]
    except:
        pass
    vector[index] = 1.
    return vector

def get_data_from_sentences(sentences, words_dict, tags_dict, ne_dict):
    all_data = []
    A = np.zeros((len(ne_dict) + 1, len(ne_dict) + 1))
    total_tokens = 0
    for sentence in sentences:
        word_data = []
        class_data = []
        tag_data = []
        words = []
        prior_entity = len(ne_dict)
        #print('sentence', sentence)
        for word, tag, entity in sentence:
            if word == _partial_word:
                continue
            words.append(word)
            word_vector = aux.get_clean_word_vector(word, tag)
            vector = word_vector
            tag_vector = get_tagging_vector(tag, tags_dict)
            tag_data.append(tag_vector)
            class_vector = get_class_vector(entity, ne_dict)
            entity_num = ne_dict[entity]
            word_data.append(vector)
            class_data.append(class_vector)
            A[prior_entity, entity_num] += 1
            prior_entity = entity_num
            total_tokens += 1
        all_data.append((words, word_data, tag_data, class_data))
    #print('A', A)
    print('total tokens', total_tokens)
    A /= total_tokens
    #print('all_data', all_data)
    return all_data, A

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

def get_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]

def create_full_sentence(words):
    import re

    sentence = ' '.join(words)
    sentence = re.sub(r' (\'[a-zA-Z])', r'\1', sentence)
    sentence = re.sub(r' \'([0-9])', r' \1', sentence)
    sentence = re.sub(r' (,.)', r'\1', sentence)
    sentence = re.sub(r' " (.*) " ', r' "\1" ', sentence)
    return sentence

def main():
    bucket_size=10
    epochs=10
    #word_emb = KeyedVectors.load_word2vec_format("pretrained-models/fasttext.4B.id.300.epoch5.uncased.vec")
    sentences = get_data("/home/asep/Documents/Ria/Github/UD_Indonesian-GSD/id_gsd-ud-dev.conllu")
    words_dict, tags_dict, ne_dict = build_dict(sentences)
    tags =[]
    for tag in tags_dict.keys():
        tags.append(tag)
    print(tags)
    #print(ne_dict)
    data, trans_prob = get_data_from_sentences(sentences, words_dict, tags_dict, ne_dict)
    #print(data[0])
    buckets = bin_data_into_buckets(data, bucket_size)
    gcn_model = GCNNerModel(dropout=0.7)
    #print(gcn_model)
    for i in range(epochs):
        random_buckets = sorted(buckets, key=lambda x: random.random())
        sys.stderr.write('--------- Epoch ' + str(i) + ' ---------\n')
        #print(random_buckets)
        for bucket in random_buckets:
            gcn_bucket = []
            for item in bucket:
                words = item[0]
                #print('words', words)
                word_embeddings = item[1]
                #print("word_embedding", word_embeddings)
                tags = item[2]
                #print("tags", tags)
                sentence = create_full_sentence(words)
                #print(sentence)
                label = item[3]
                #print("label", label)
                label = [np.array(l, dtype=np.float32) for l in label]
                A_fw, A_bw, _, X = aux.create_graph_from_sentence_and_word_vectors(sentence, word_embeddings)
                print("XX", X.shape)
                gcn_bucket.append((A_fw, A_bw, X, tags, label))
            if len(gcn_bucket) > 1:
                gcn_model.train(gcn_bucket, trans_prob, 1)
if __name__ == '__main__':
    main()

