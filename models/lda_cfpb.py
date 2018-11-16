#!bin/bash/python

import gzip
import numpy as np
import pandas as pd
import random
import string
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
tqdm.pandas(desc='progress-bar')
from gensim import corpora, models
from sklearn import model_selection
from sklearn import cross_validation
from tabulate import tabulate
import pyLDAvis.gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
from sklearn import svm


api = "https://data.consumerfinance.gov/resource/jhzv-w97w.json"
query = '?&$limit=1000000'
#$where=date%20between%20%272014-01-01T00:00:00%27%20and%20%272015-01-01T00:00:00%27'

dataset_identifier = 'jhzv-w97w'
APP_TOKEN = '48ozcpj4nCO3mqgJOl8GoIJgF'
token = '?$$app_token='
full_query = api+query
cfpb = pd.read_json(full_query)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(cfpb['product'])
cfpb['le_product'] = le.transform(cfpb['product'])

data = cfpb[['le_product', 'complaint_what_happened', 'date_received','company']]
data = data.dropna()
data['product'] = data['le_product'].map(int)
print('{} complaints'.format(len(data)))
data = data[['complaint_what_happened','product','date_received','company']]

additional_stop_words = ['xxx','xxxx','00','xx','xx xx','000','00and','xxxxxxxx']
additional_stop_words = [unicode(i,'utf-8') for i in additional_stop_words]
stop = set(stopwords.words('english') + list(string.punctuation) + additional_stop_words)
stemmer = PorterStemmer()
re_punct = re.compile('[' + ''.join(string.punctuation) + ']')

def preprocess(text):
    try:
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if not t in stop]
        tokens = [re.sub(re_punct, '', t) for t in tokens]
        tokens = [re.sub(r'\d','',t) for t in tokens] # there are random digits in this dataset
        tokens = [re.sub(r'xx+','',t) for t in tokens] # there are also random xs all through it for anonymization
        tokens = [t for t in tokens if len(t) > 2]
        tokens = [stemmer.stem(t) for t in tokens]
        if len(tokens) == 0:
            return None
        else:
            return ' '.join(tokens)
    except:
        return None

data['tokens'] = data['complaint_what_happened'].progress_map(preprocess)

##############

data = data[data['tokens'].notnull()]
print data.shape
data.reset_index(inplace=True)
data.drop('index', inplace=True, axis=1)
print('{} complaints'.format(len(data)))

texts = [tokens.split() for tokens in data.tokens]
id2word = corpora.Dictionary(texts)
corpus = [id2word.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('complaints_full.mm', corpus)
id2word.save('complaints_full.dict')

############

def ldaCV(n_topics, corpus, id2word, test_size=0.2, sample=None):
    if not isinstance(n_topics, list):
        n_topics = [n_topics]
    obs = len(corpus)
    corpus = np.array(corpus) # draws from corpus created above

    if sample is not None:
        if sample < 1:
            sample_idx = random.sample(range(obs), int(obs*sample)) # inital pass assigns each sample to a random
        else:
            sample_idx = random.sample(range(obs), int(sample))
        corpus = corpus[sample_idx]

    train, test = cross_validation.train_test_split(corpus, test_size=test_size) # cross validated
    perplexities = []

    for n in n_topics:
        print('{} topics'.format(n))
        model = models.ldamodel.LdaModel(train, num_topics=n, id2word=id2word)
        perplexity = model.log_perplexity(test)
        print(' - Perplexity: {}'.format(round(perplexity, 3)))
        perplexities.append((n, perplexity))

    return perplexities

%time p = ldaCV(list(range(50,450,50)),corpus,id2word)

x, y = zip(*p)
plt.plot(x, y)
plt.scatter(x, y)
plt.show()

############

model = models.ldamodel.LdaModel(corpus, num_topics=250, id2word=id2word)
model.save('complaints_full_lda.model')
new_id2word = corpora.Dictionary()
_ = new_id2word.merge_with(id2word) # this doesn't return us anything of use right now
id2word = corpora.Dictionary.load('complaints_full.dict')
corpus = corpora.MmCorpus('complaints_full.mm')
model = models.ldamodel.LdaModel.load('complaints_full_lda.model')

##########

complaints_vis = pyLDAvis.gensim.prepare(model, corpus, id2word)
pyLDAvis.display(complaints_vis)

##########

pyLDAvis.save_html(complaints_vis, 'complaints_full_lda_graphic.html')

#########

# Doc2Vec
sent_data = data.copy()
y = []
doc_vectors = []
for i, row in sent_data.iterrows():
    doc_vectors.append(TaggedDocument(row['tokens'].split(), ['doc_' + str(i)]))
    y.append(row['product'])

def shuffle_docs(docs):
    random.shuffle(docs)
    return docs

model = Doc2Vec(size=100, window=10, min_count=1, workers=4)
model.build_vocab(doc_vectors)
for epoch in range(30):
    print('Epoch {}'.format(epoch)),
    model.train(shuffle_docs(doc_vectors))
d2v = {d: vec for d, vec in zip(model.docvecs.offset2doctag, model.docvecs.doctag_syn0)}
X = []
for d in range(len(doc_vectors)):
    X.append(d2v['doc_' + str(d)])
X = np.array(X)

model.save('complaints_full_doc2vec.model')
model = Doc2Vec.load('complaints_full_doc2vec.model')

d2v = {d: vec for d, vec in zip(model.docvecs.offset2doctag, model.docvecs.doctag_syn0)}
X = []
for d in range(len(doc_vectors)):
    X.append(d2v['doc_' + str(d)])
X = np.array(X)

########

model.most_similar('titl')

#######

# Classification
# Multi-Class
x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

# Linear Model
classifier = svm.LinearSVC()
classifier.fit(x_train, y_train)
print('Accuracy: {}'.format(round(classifier.score(x_test, y_test), 3)))

######

# RBF Model
from sklearn.svm import SVC
clf = svm.SVC(kernel='rbf',C=10)
clf.fit(x_train,y_train)
print('Accuracy: {}'.format(round(clf.score(x_test, y_test), 3)))



########
Y = y
#Waterfall
y = [1 if i==7 else 0 for i in Y]
