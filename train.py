from __future__ import unicode_literals, print_function, division
import argparse

from gcn_ner import GCNNer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GInNER: Graph-based Indonesian Named Entity Recognition ')
    parser.add_argument("--n_epoch", type=int, default=101, help="train model in total n epochs")
    parser.add_argument("--dataset", type=str, default="", help="Dataset path")
    parser.add_argument("--saving_dir", type=str, default="", help="Saving models directory path")
    GCNNer.train_and_save(dataset='./datasets/dev.conllu', saving_dir='./datasets/', epochs=201)
