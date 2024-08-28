#!/usr/bin/env python
# coding: utf-8

# In[29]:

import gensim
from gensim.models import Word2Vec
import numpy #numpy version 1.17.4 needed to load model correctly
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))

cars_wv = os.path.join(__dir__, 'cars_wv.model')
lead_classifier = os.path.join(__dir__, 'lead_classifier.h5')

numpy.version.version
wv_model = Word2Vec.load(cars_wv)
nn_model = load_model(lead_classifier)
MAX_SEQUENCE_LENGTH = 46


def get_chat_type(chat_query): 
    query = gensim.utils.simple_preprocess(chat_query)
    query_seq = []
    for word in query:
        if word in wv_model.wv.vocab:
            word_index = wv_model.wv.vocab[word].index
            query_seq.append(word_index)
        else:
            query_seq.append(0)
    query_seq = pad_sequences([query_seq],maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    y_pred = nn_model.predict(numpy.asarray(query_seq))
    y_val = y_pred.tolist()[0]
    ix = y_val.index(max(y_val))
    chat_types = ['other', 'parts', 'sales', 'service']
    return chat_types[ix]

