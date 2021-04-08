from __future__ import unicode_literals, print_function, division
import argparse

from gcn_ner import GCNNer

def main(train_dataset, validation_dataset, saving_dir, mode, n_epochs, ner_model):
    if mode=="train":
        GCNNer.train_and_save(train_dataset, validation_dataset, saving_dir, n_epochs)
    
    if mode=="test":
        ner = GCNNer(ner_filename=ner_model, trans_prob_file='./datasets/trans_prob.pickle')
        ner.test('./datasets/test.conllu')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GInNER: Graph-based Indonesian Named Entity Recognition ')
    parser.add_argument("--n_epochs", type=int, default=101, help="train model in total n epochs")
    parser.add_argument("--train_dataset", type=str, default="", help="Train dataset path")
    parser.add_argument("--validation_dataset", type=str, default="", help="Validation dataset path")
    parser.add_argument("--saving_dir", type=str, default="", help="Saving models directory path")
    parser.add_argument("--mode", type=str, default="train", help="use which mode type: train/test")
    parser.add_argument("--ner_model", type=str, default="./datasets/ner-gcn-100.tf", help="use which mode type: train/test")
    
    args = parser.parse_args()
    main(args.train_dataset, args.validation_dataset, args.saving_dir, args.mode, args.n_epochs, args.ner_model)
