import logging
from sklearn.metrics import recall_score, precision_score, f1_score

_logger = logging.getLogger(__name__)


def get_gcn_results(gcn_model, data, trans_prob):
    import numpy as np
    import copy

    from ..aux import create_graph_from_sentence_and_word_vectors
    from ..aux import create_full_sentence
    from ..aux import tags

    TAGS = copy.deepcopy(tags)
    TAGS.append('UNK')

    total_sentences = 0
    broken_sentences = 0
    correct_sentences = 0
    recall_scores = []
    precision_scores = []
    f1_scores = []
    for words, sentence, tag, classification in data:
        full_sentence = create_full_sentence(words)
        word_embeddings = sentence
        total_sentences += 1
        try:
            A_fw, A_bw, tags, X = create_graph_from_sentence_and_word_vectors(full_sentence, word_embeddings)
            prediction = gcn_model.predict_with_viterbi(A_fw, A_bw, X, tags, trans_prob)
            correct_sentences += 1
        except Exception as e:
            _logger.warning('Cannot process the following sentence: ' + full_sentence)
            broken_sentences += 1
            continue

        y_true = classification
        y_pred = prediction
        recall_scores.append(recall_score(y_true, y_pred, average='macro'))
        precision_scores.append(precision_score(y_true, y_pred, average='macro'))
        f1_scores.append(f1_score(y_true, y_pred, average='macro'))
            
    print("total sentences", total_sentences)
    print("correct sentences", correct_sentences)
    print("broken sentences", broken_sentences)
    #print("precision", precision_scores)
    #print("recall", recall_scores)
    #print("f scores", f1_scores)
    avg_recall = sum(recall_scores)/len(recall_scores)
    avg_precision = sum(precision_scores)/len(precision_scores)
    avg_fscores = sum(f1_scores)/len(f1_scores)
    print("precision {}, recall {}, f1 {}".format(avg_precision, avg_recall, avg_fscores))
    return avg_precision, avg_recall, avg_fscores
