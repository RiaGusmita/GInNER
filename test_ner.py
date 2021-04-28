_error_message = '''
Please provide a text as an input.
You can either provide the text as an argument: python test_ner.py Hard to believe this program was made in September 2017.
Or pipe the text from the command line: python test_ner.py < data/random_text.txt
'''
import torch
import os.path as path
from data_loader import get_words_embeddings_from_sentence, create_graph_from_sentence_and_word_vectors, get_class_name, get_words_embeddings_from_sentence_fasttext
import argparse

DEVICE = torch.device("cpu")

def _aggregate_sentence(args):
    return_str = ''
    for argument in args:
        return_str += argument + ' '
    return return_str


def _get_entity_tuples_from_sentence(sentence, word_emb_model, word_embedding_dim):
    from model import GInNER
    import fasttext
    import fasttext.util
    word_emb=""
    if word_emb_model=="fasttext":
        word_emb = fasttext.load_model('cc.id.300.bin')
        word_emb = fasttext.util.reduce_model(word_emb, word_embedding_dim)
        
    ginner = GInNER(word_embedding_dim, DEVICE, dropout=0.0, hidden_layer=2, nheads=3)
    checkpoint = torch.load(path.join("models", "checkpoint_epoch_{}.pt".format(49)))
    ginner.load_state_dict(checkpoint["model_state_dict"])
    ginner.to(DEVICE)
    
    if word_emb_model=="fasttext":
        vectors = get_words_embeddings_from_sentence_fasttext(sentence, word_emb)
    else:
        words, vectors, idx = get_words_embeddings_from_sentence(sentence)
    
    word_embeddings=vectors
    A, X = create_graph_from_sentence_and_word_vectors(sentence, word_embeddings)
    output_tensor = ginner(X, A)
    output_tensor = output_tensor
    logits_scores, logits_tags = torch.max(output_tensor, 1, keepdim=True)
    logits_label = logits_tags.detach().cpu().numpy().tolist()
    y_pred = [predict[0] for predict in logits_label]
    for i, word in enumerate(words):
        print("word {} prediction {}".format(word, get_class_name(y_pred[i])))

if __name__ == '__main__':
    import os
    import sys
    
    parser = argparse.ArgumentParser(description='Indonesian NER')
    parser.add_argument("--word_emb_model", type=str, default="spacy", help="Word embedding models: spacy/fasttext")
    parser.add_argument("--word_emb_dim", type=int, default=96, help="Word embedding dimension")
    args = parser.parse_args()
    if os.isatty(0):
        print(_error_message)
        exit(0)
    sentences = sys.stdin.read().strip()
    sentences = sentences.split("\n")
    for sentence in sentences:
        print(_get_entity_tuples_from_sentence(sentence, args.word_emb_model, args.word_emb_dim))
