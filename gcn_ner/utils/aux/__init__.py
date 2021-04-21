import spacy
import re
import numpy as np
from tqdm import tqdm

parser = spacy.load("id_spacy")


default_vector = parser('entity')[0].vector

tags = ['PROPN', 'NOUN', 'VERB', 'ADP', 'ADJ', 'PUNCT', 'CCONJ', 'PRON', 'ADV', 'DET', 'SCONJ', 'AUX', 'PART', 'NUM', 'SYM', 'X', 'CONJ']
'''
        'NSD', 'VSA', 'R--', 'Z--', 'H--', 'D--', 'S--', 'F--', 'VSP', 'G--', 'ASP', 'X--', 'B--', 'M--', 
        'PP3', 'O--', 'VSA+PS3', 'CC-', 'CO-', 'F--+PS3', 'G--+T--', 'W--', '_', 'W--+T--', 'ASP+PS3', 'NSD+PS3', 
        'VSA+T--', 'PS2', 'VSP+PS3', 'T--', 'PS3', 'R--+PS3', 'NSF', 'PS1+VSA+T--', 'PS1', 'NPD', 'M--+T--', 'R--+PS1', 
        'ASS', 'H--+T--', 'R--+PS2', 'PP1', 'ASP+T--', 'CD-', 'D--+T--', 'B--+PS3', 'NSD+PS1', 'NPD+PS3', 'NSM', 'D--+PS3', 
        'PS1+VSA', 'I--', 'APP', 'VSP+T--', 'NSD+T--', 'NSD+PS2', 'CO-+PS3', 'B--+T--', 'CC-+PS3', 'PP2', 'M--+PS3']
'''
classes = ["O", "B-PERSON", "I-PERSON", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]

word_substitutions = {'-LRB-': '(',
                      '-RRB-': ')',
                      '``': '"',
                      "''": '"',
                      "--": '-',
                      }
_partial_word = '%pw'


def create_full_sentence(words):
    import re

    sentence = ' '.join(words)
    sentence = re.sub(r' (\'[a-zA-Z])', r'\1', sentence)
    sentence = re.sub(r' \'([0-9])', r' \1', sentence)
    sentence = re.sub(r' (,.)', r'\1', sentence)
    sentence = re.sub(r' " (.*) " ', r' "\1" ', sentence)
    sentence = sentence.replace('do n\'t', 'don\'t')
    sentence = sentence.replace('did n\'t', 'didn\'t')
    sentence = sentence.replace('was n\'t', 'wasn\'t')
    sentence = sentence.replace('were n\'t', 'weren\'t')
    sentence = sentence.replace('is n\'t', 'isn\'t')
    sentence = sentence.replace('are n\'t', 'aren\'t')
    sentence = sentence.replace('\' em', '\'em')
    sentence = sentence.replace('s \' ', 's \'s ')
    return sentence


def clean_word(word, tag):
    word = word
    if tag == '.':
        word = word.replace('/', '')
    if word in word_substitutions:
        word = word_substitutions[word]
    word = re.sub(r'\+([a-zA-Z])', r'\1', word)
    word = re.sub(r'\=([a-zA-Z])', r'\1', word)
    word = re.sub(r'([a-zA-Z]+)_([a-zA-Z]+)', r' \1-\2 ', word)
    return word


def vector_is_empty(vector):
    to_throw = 0
    for item in vector:
        if item == 0.0:
            to_throw += 1
    if to_throw == len(vector):
        return True
    return False


def get_clean_word_vector(word, tag):
    parsed = parser(clean_word(word, tag))
    try:
        vector = parsed[0].vector
        if vector_is_empty(vector):
            vector = default_vector
    except:
        vector = default_vector
    #print("vector", vector.shape)
    return np.array(vector, dtype=np.float64)


def get_tagging_vector(tag):
    vector = [0.] * (len(tags) + 1)
    index = len(tags)
    try:
        index = tags.index(tag)
    except:
        pass
    vector[index] = 1.
    return vector


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


def _get_word_vectors_from_tokens(tokens):
    words = []
    vectors = []
    idx = []
    for token in tokens:
        word = token.orth_
        tag = token.tag_
        words.append(clean_word(word, tag))
        vectors.append(get_clean_word_vector(word, tag))
        idx.append([token.idx, token.idx + len(token.orth_)])
    return words, vectors, idx


def get_entity_name(prediction):
    index = np.argmax(prediction)
    try:
        return classes[index]
    except:
        return ''


def get_words_embeddings_from_sentence(sentence):
    sentence = re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r' \1_\2 ', sentence)
    tokens = parser(sentence)
    return _get_word_vectors_from_tokens(tokens)


def get_words_embeddings_from_text(text):
    doc = parser(text)
    sentences = []
    for sent in doc.sents:
        sentences.append(_get_word_vectors_from_tokens(sent))
    return sentences


def create_graph_from_sentence_and_word_vectors(sentence, word_vectors):
    if not isinstance(sentence, str):
        raise TypeError("String must be an argument")
    from igraph import Graph
    from .nl import SpacyTagger as Tagger, SpacyParser as Parser

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
    A_fw = np.array(g.get_adjacency().data)

    nodes, edges, words, tags, types = parser.execute_backward()
    g2 = Graph(directed=True)
    g2.add_vertices(nodes)
    g2.add_edges(edges)
    A_bw = np.array(g2.get_adjacency().data)
    #print(tags)
    #print(len(tags))
    #print(src_tags)
    #print(len(src_tags))
    #print(len(tags))
    tag_logits = [get_tagging_vector(tag) for tag in tags]
    return A_fw, A_bw, tag_logits, X


def get_all_sentences(filename):
    file = open(filename)
    sentences = []
    items = []
    #old_entity = ''
    for line in file.readlines():
        elements = line.split()
        #print(elements)
        #print(len(elements))
        if len(elements) == 0:
            if items != []:
                #print('items', items)
                sentences.append(items)
            items = []
            continue
        else:
            word = elements[2].strip()
            #print("word", word)
            tag = elements[3].strip()
            #print('tag', tag)
            entity = elements[9]
            #print('entity', entity)
            entity = re.sub("\n", "", entity)
            if entity =="_":
                entity="O"
            items.append((word, tag, entity))
            #print(items)
    return sentences


def decide_entity(string, prior_entity):
    if string == '*)':
        return prior_entity, ''
    if string == '*':
        return prior_entity, prior_entity
    entity = ''
    for item in classes:
        if string.find(item) != -1:
            entity = item
    prior_entity = ''
    if string.find(')') == -1:
        prior_entity = entity
    return entity, prior_entity


def get_data_from_sentences(sentences):
    all_data = []
    A = np.zeros((len(classes) + 1, len(classes) + 1))
    total_tokens = 0
    for sentence in tqdm(sentences):
        word_data = []
        class_data = []
        tag_data = []
        tag_text = []
        words = []
        prior_entity = len(classes)
        #print('sentence', sentence)
        for word, tag, entity in sentence:
            if word == _partial_word:
                continue
            words.append(clean_word(word, tag))
            word_vector = get_clean_word_vector(word, tag)
            tag_vector = get_tagging_vector(tag)
            tag_data.append(tag_vector)
            tag_text.append(tag)
            vector = word_vector
            class_vector = get_class_vector(entity)
            entity_num = get_entity_num(entity)
            word_data.append(vector)
            class_data.append(class_vector)
            A[prior_entity, entity_num] += 1
            prior_entity = entity_num
            total_tokens += 1
        all_data.append((words, word_data, tag_data, class_data))
    #print('A', A)
    #print('total tokens', total_tokens)
    A /= total_tokens
    #print('all_data', all_data)
    return all_data, A
