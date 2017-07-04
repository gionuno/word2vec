#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 13:52:13 2017

@author: quien
"""

import numpy as np;
import numpy.random as rd;
import matplotlib.pyplot as plt;

def sigm(t):
    return 1.0/(1.0+np.exp(-t));

def J(x,Y,t):
    s = sigm(t*np.dot(Y.T,x));
    return -np.mean(np.log(s));

def dJ(x,Y,t):
    s  = sigm(t*np.dot(Y.T,x));
    dw = -np.outer(t*(1.0-s),x).T/t.shape[0];
    dx = -np.dot(t*(1.0-s),Y.T)/t.shape[0];
    return dw,dx;

class word2vec:
    def __init__(self,U,G,D):
        self.U = U; #dict of lists of 2K words
        self.G = G; #dict of adyacency graph for sampling
        self.D = D; #dimen
        self.V = len(G);
        
        self.X  = rd.randn(self.D,self.V)/np.sqrt(self.V); 
        self.dX = np.zeros(self.X.shape);
        self.mX = np.zeros(self.X.shape);
        
        self.Y  = rd.randn(self.D,self.V)/np.sqrt(self.V);
        self.dY = np.zeros(self.Y.shape);
        self.mY = np.zeros(self.Y.shape);
        
        self.IT = 10;
        self.it = 0;
    
    def get_tup(self,u):
        r = rd.randint(len(self.U[u]));
        return self.U[u][r];
    
    def get_contexts(self,u,K):
        w = [];
        t = [];
        for k in range(K):
            r = rd.randint(self.V);
            while r in self.G[u] or u in self.G[r] or r == u:
                r = rd.randint(self.V);
            
            s = rd.randint(len(self.U[r]));
            for v in self.U[r][s]:
                w.append(v);
                t.append(-1.0);
        for k in range(K):
            r = rd.randint(len(self.U[u]));
            for v in self.U[u][r]:
                w.append(v);
                t.append(1.0);
        return np.array(t),np.array(w);
    
    def step(self,B,K,dt,mu):
        mmX = np.zeros(self.mX.shape);
        mmY = np.zeros(self.mY.shape);
        for u in range(self.V):
            mf  = 0.0;
            for b in range(B):
                t,w = self.get_contexts(u,K);
            
                xu = self.X[:,u]+dt*mu*self.mX[:,u];
                Yw = self.Y[:,w]+dt*mu*self.mY[:,w];
                
                f = J(xu,Yw,t);
                mf += f/B;
                
                dYw,dxu = dJ(xu,Yw,t);
                self.mX[:,u] = mu*self.mX[:,u]-dt*dxu/B;
                self.mY[:,w] = mu*self.mY[:,w]-dt*dYw/B;
                
                self.X[:,u] += self.mX[:,u];
                self.Y[:,w] += self.mY[:,w];
                
            print u,mf;
            