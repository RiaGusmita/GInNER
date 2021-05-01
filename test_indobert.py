#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import BertTokenizer, AutoModel
import torch
import numpy as np

def word_subword_tokenize(sentence, tokenizer):
    # Add CLS token
    subwords = []#[tokenizer.cls_token_id]
    subword_to_word_indices = []#[-1] # For CLS

    # Add subwords
    print("sentence length", len(sentence.split(" ")))
    for word_idx, word in enumerate(sentence.split(" ")):
        #print("word", word)
        subword_list = tokenizer.encode(word, add_special_tokens=False)
        print("word", word, "subword_list", subword_list, "word_idx", word_idx)
        if len(subword_list)>1:
            print(round(np.average(subword_list)))
            subword_list = [round(np.average(subword_list))]
        subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            
        subwords += subword_list

    return subwords, subword_to_word_indices

tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
text = "Indonesia"
subwords, subword_to_word_indices = word_subword_tokenize(text, tokenizer)

subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)
print(subwords.shape)
subword_to_word_indices = torch.LongTensor(subword_to_word_indices).view(1, -1).to(model.device)
logits = model(subwords, subword_to_word_indices)[0]

print(logits.squeeze())
print(logits.squeeze().shape)

