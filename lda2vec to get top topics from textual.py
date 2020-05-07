import pandas as pd 
import numpy as np
np.random.seed(2018)

import math 
import gzip 
import logging
import re 
import string 
from tqdm import tqdm 
from pprint import pprint

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *




data = pd.read_excel ('RFactors_2015-19.xlsx')
#documents = data.risk_factors
#data_text['ticker'] = data.ticker
data_text = data[['risk_factors']]
data_text['index'] = data_text.index
data_text['ticker'] = data.ticker
documents = data_text

print(len(documents))
print(documents[:5])

# lemmatization, stemming, tokenize
def lemmatize_stemming(text):
    stemmer=SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
    
 # test sample
doc_sample = documents[documents['index'] == 23000].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
sample=preprocess(doc_sample)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

# process 
processed_docs = documents['risk_factors'].fillna('').astype(str).map(preprocess)
processed_docs[:10]  ## show 10 doc

# BOW (bag-of-word) method
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 5:
        break

# filter
## keep tokens >5 && < overall/2 documents
dictionary.filter_extremes(no_above=0.5, keep_n=3000)
## resulting dict: 3000 unique tokens
## include more unique tokens than previous after filter
## choose this one


# doc2bow
## (number of word in this document, number of times these words appear)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[3000] ## take 3000 documents from total 3056 availible docs

## preview of BOW
bow_doc_3000 = bow_corpus[3000]
for i in range(len(bow_doc_3000)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_3000[i][0], 
                                               dictionary[bow_doc_3000[i][0]], 
bow_doc_3000[i][1]))



# running LDA using BOW
lda_model = gensim.models.LdaMulticore(bow_corpus,num_topics=10, 
                                       id2word=dictionary, passes=10, 
                                       workers=2,minimum_probability=0)
## passes: # of laps the model will take through the corpus, more the accurate

## each topic: word occurense & its relative weight
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

## able to distinguish different topics using the words & its weight in each topic 


## gives the percentage of each topic above, overall document topics
for i in range(len(processed_docs)):
    print(lda_model[bow_corpus[i]])
