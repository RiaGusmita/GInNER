def get_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


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


def train_and_save(train_dataset, validation_dataset, saving_dir, epochs=20, bucket_size=10):
    import random
    import sys
    import pickle
    import numpy as np
    import gcn_ner.utils.aux as aux
    from pathlib import Path

    from gcn_ner.ner_model import GCNNerModel
    from ..aux import  create_full_sentence
    from ..testing import get_gcn_results
    from tqdm import tqdm
    import visdom
    viz = visdom.Visdom()

    sentences = aux.get_all_sentences(train_dataset)
    val_sentences = aux.get_all_sentences(validation_dataset)
    print('Computing the transition matrix')
    data, trans_prob = aux.get_data_from_sentences(sentences)
    val_data, val_trans_prob = aux.get_data_from_sentences(val_sentences)
    #my_file = Path("./datasets/trans_prob.pickle")
    #if my_file.exists():
    #    pickle.dump(trans_prob, open("./datasets/trans_prob.pickle", "wb"))
    #else:
    #    pickle.dump(trans_prob, open("./datasets/trans_prob.pickle", "wb"))
        
    #print(data)
    buckets = bin_data_into_buckets(data, bucket_size)
    val_buckets = bin_data_into_buckets(val_data, bucket_size)
    gcn_model = GCNNerModel(dropout=0.7)
    viz.line([[0.0], [0.0]], [0.], win='{}_loss'.format("train"), opts=dict(title='train loss', legend=['train loss', 'validation loss']))
    for i in range(epochs):
        random_buckets = sorted(buckets, key=lambda x: random.random())
        sys.stderr.write('--------- Epoch ' + str(i) + ' ---------\n')
        total_loss = 0
        for bucket in tqdm(random_buckets):
            try:
                gcn_bucket = []
                x=1
                for item in bucket:
                    words = item[0]
                    word_embeddings = item[1]
                    sentence = create_full_sentence(words)
                    if len(sentence) > 250 and x > 2:
                        pass
                    else:
                        #print("sentence length", len(sentence))
                        #print(sentence)
                        
                        tags = item[2]
                        label = item[3]
                        label = [np.array(l, dtype=np.float32) for l in label]
                        A_fw, A_bw, _, X = aux.create_graph_from_sentence_and_word_vectors(sentence, word_embeddings)
                        gcn_bucket.append((A_fw, A_bw, X, tags, label))
                    x += 1
                if len(gcn_bucket) > 1:
                    #print("train bucket")
                    loss = gcn_model.train(gcn_bucket, trans_prob, 1)
                    total_loss +=loss
            except:
               pass
        save_filename = saving_dir + '/ner-gcn-' + str(i) + '.tf'
        print(save_filename)
        sys.stderr.write('Saving into ' + save_filename + '\n')
        gcn_model.save(save_filename)
        
        val_random_buckets = sorted(val_buckets, key=lambda x: random.random())   
        total_val_loss = 0
        for val_bucket in tqdm(val_random_buckets):
            try:
                gcn_val_bucket = []
                for item in val_bucket:
                    words = item[0]
                    #print('words', words)
                    word_embeddings = item[1]
                    sentence = create_full_sentence(words)
                    tags = item[2]
                    #print("tags", tags)
                    label = item[3]
                    label = [np.array(l, dtype=np.float32) for l in label]
                    A_fw, A_bw, _, X = aux.create_graph_from_sentence_and_word_vectors(sentence, word_embeddings)
                    gcn_val_bucket.append((A_fw, A_bw, X, tags, label))
                if len(gcn_val_bucket) > 1:
                    loss = gcn_model.train(gcn_val_bucket, val_trans_prob, 1)
                    total_val_loss +=loss
            except:
               pass
        total_loss = total_loss/len(random_buckets)
        total_val_loss = total_val_loss/len(val_random_buckets)
        #print("total_loss", total_loss)
        viz.line([[total_loss, total_val_loss]], [i], win='{}_loss'.format("train"), update='append')
        
        #total_loss = total_loss/len(random_buckets)
        #print("avg loss", total_loss)
    
    return gcn_model
