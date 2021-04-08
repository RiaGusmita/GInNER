# Graph-based Indonesian Named Entity Recognition (GInNER) 
This model is used to predict named entities in Indonesian text utilizing a graph representation technique, graph convolutional network (GCN). 
The model is built by modifying an [existing GCN-NER](https://github.com/ContextScout/gcn_ner) by A. Cetoli, S. Bragaglia, A.D. O'Harney, M. Sloan (https://arxiv.org/abs/1709.10053).

## Indonesian POS tagger Model using Spacy
We utilize a pretrained Indonesian POS tagger from https://github.com/jeannefukumaru/id_dep_ud_sm.
Follow the instructions to install the model.

## Visualization Tools

We use a third party to visualize the training and validation loss, and accuracy. 
If you have not had visdom yet, install it using this command:
```
pip install visdom
``` 

Run visdom before you train the model by typing ```visdom``` on terminal and enter.
