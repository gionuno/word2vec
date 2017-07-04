#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 22:07:52 2017

@author: quien
"""

import numpy as np;
import numpy.random as rd;
import matplotlib.pyplot as plt;

from word2vec import *;

import io;
import re;
import os;

import glob;
import string;

def cleantxt(T,spaces):
    regex = re.compile("[%s]"%re.escape('!"#$%&\'()*,+-/:;<=>?@[\\]^_`{|}~'+string.digits));
    
    aux = regex.sub(" ",T);
    aux = re.sub("\s+"," ",aux).strip().lower().split('.');
    return [a.split() for a in aux if len(a) > 0];

books = glob.glob("books/*.txt");

num_words = 0;
idx2word = [];
idx2cont = [];
word2idx = {};
idxs_total = [];

for book in books:
    with io.open(book,'r') as file_:
        b = file_.read().encode('ascii','ignore');
        text = cleantxt(b,True);
        for sentence in text:
            idxs = [];
            for w in sentence:
                if w not in word2idx:
                    idx2word.append(w);
                    idx2cont.append(0);
                    word2idx[w] = num_words;
                    num_words += 1;
                idx2cont[word2idx[w]] += 1;
                idxs.append(word2idx[w]);
            idxs_total.append(idxs);

N = num_words;
D = 375;
K = 2;

U = {n:[] for n in range(N)};
G = {n:[] for n in range(N)};

for idxs in idxs_total:
    L = len(idxs);
    for l in range(L):
        u  = idxs[l];
        vs = idxs[max(l-K,0):l]+idxs[l+1:min(l+K+1,L)];
        U[u].append(vs);
        for v in vs:
            if v not in G[u]:
                G[u].append(v);
            if u not in G[v]:
                G[v].append(u);

w2v = word2vec(U,G,D);

it = 0;
while it < 15:
    print it;
    it += 1;
    w2v.step(10,2,1e-6,1e-4);

np.savetxt('w2vX.txt',w2v.X);
np.savetxt('idx2word.txt',idx2word,fmt="%s");
np.savetxt('idx2cont.txt',idx2cont);
a = np.argsort(-np.array(idx2cont));

l = a[:1000:2];
X = w2v.X[:,l];

XtX = np.dot(X.T,X);
dX = np.diag(XtX);

D2 = np.outer(np.ones(XtX.shape[0]),dX)+np.outer(dX,np.ones(XtX.shape[0]))-2.0*XtX;
J = np.eye(XtX.shape[0])-np.ones((XtX.shape[0],XtX.shape[0]))/XtX.shape[0];

E,V = np.linalg.eig(-J*D2*J);

f = plt.scatter(V[:,0],V[:,1],marker='o',s=3,c=range(len(l)),cmap='hot');
for w,x,y in zip(l,V[:,0],V[:,1]):
    plt.annotate(idx2word[w],xy=(x,y))
plt.show();