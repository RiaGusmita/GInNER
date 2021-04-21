#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 23:42:42 2021

@author: asep
"""

import spacy
id = spacy.load("id_spacy")

doc = id("Bantulah kami melengkapi berita dan informasi")

for token in doc:
    print("text", token.text)
    print("orth", token.orth_)
    print("pos", token.pos_)
    print("type", token.dep_)
    print("token children")
    for l in token.children:
        print(l)
        print(l.orth_)
    print("######")
    print("")