# Graph-based Indonesian Named Entity Recognition (GInNER)
This model is used to predict named entities in Indonesian text utilizing a graph representation technique, graph attention network (GAT). It is also equipped with several word embeddings models including Spacy, fastText, fastText-IndoBERT, and IndoBERT.
  
## Datasets
The model uses datasets provided by IndoNLU: https://github.com/indobenchmark/indonlu

## Training Dataset Format
Training dataset is in BIO format where the first column refers to words and the second column represents named entity types. These are some examples:
```
Produser	O
David	B-PERSON
Heyman	I-PERSON
dan	O
sutradara	O
Mark	B-PERSON
Herman	I-PERSON
sedang	O
mencari	O
seseorang	O
yang	O
mampu	O
memerankan	O
tokoh	O
utama	O
yang	O
lugu	O
,	O
sehingga	O
mereka	O
meminta	O
setiap	O
anak	O
apa	O
yang	O
mereka	O
tahu	O
tentang	O
Holocaust	O
.	O
```
