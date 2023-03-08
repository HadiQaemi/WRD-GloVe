import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from scipy.optimize import linprog

embedding_dim = 100

sen_1 = 'Obama speaks to the media in illinois'
sen_2 = 'the president speaks the press in chicago'

#Head to https://nlp.stanford.edu/projects/glove/ (where you can learn more about the GloVe algorithm), and download the pre-computed embeddings from 2014 English Wikipedia.
glove_dir = 'GLOVE_DIRECTION'

#Let's parse the un-zipped file (it's a txt file) to build an index mapping words (as strings) to their vector representation (as number vectors).
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


def EMD(p, q, D):
    
    # wasserstein distance: earth mover’s distance (EMD) 
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = D.reshape(-1)
    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    return result.fun

def WRD(x, y): # word rotator distance
    
    # Determined Norm and Direction of Word Vectors
    x_norm = (x**2).sum(axis=1, keepdims=True)**0.5
    y_norm = (y**2).sum(axis=1, keepdims=True)**0.5
    p = x_norm[:, 0] / x_norm.sum()
    q = y_norm[:, 0] / y_norm.sum()
    D = 1 - np.dot(x / x_norm, (y / y_norm).T)
    wrd = EMD(p, q, D)
    
    return 1 - wrd # word rotator similarity

vectors_sen1 = text_to_word_sequence(sen_1)
embedding_vector_sen1 = np.zeros((len(vectors_sen1), embedding_dim))

vectors_sen2 = text_to_word_sequence(sen_2)
embedding_vector_sen2 = np.zeros((len(vectors_sen2), embedding_dim))

for i in range(len(vectors_sen1)):
  embedding_vector_sen1[i] = embeddings_index.get(vectors_sen1[i])


for i in range(len(vectors_sen2)):
  embedding_vector_sen2[i] = embeddings_index.get(vectors_sen2[i])

similarity = WRD(embedding_vector_sen1, embedding_vector_sen2)
print('word_rotator_similarity is', similarity)