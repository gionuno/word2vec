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
        
        self.X  = rd.randn(self.D,self.V)/np.sqrt(self.D); 
        self.mX = np.zeros(self.X.shape);
        
        self.Y  = rd.randn(self.D,self.V)/np.sqrt(self.D);
        self.mY = np.zeros(self.Y.shape);
        
        self.IT = 10;
        self.it = 0;
    
    def get_tup(self,u):
        r = rd.randint(len(self.U[u]));
        return self.U[u][r];
    
    def get_contexts(self,u,K):
        w = [ u ];
        t = [1.0];
        for k in range(K):
            r = rd.randint(self.V);
            while r in self.G[u] or u in self.G[r] or r == u:
                r = rd.randint(self.V);
            w.append(  r );
            t.append(-1.0);
        r = rd.randint(len(self.U[u]));
        return np.array(self.U[u][r]),np.array(t),np.array(w);
    
    def step(self,B,K,dt,mu):
        for u in range(self.V):
            mf  = 0.0;
            for b in range(B):                
                v,t,w = self.get_contexts(u,K);
                
                xu = np.mean(self.X[:,v],1);
                Yw = self.Y[:,w];
                f = J(xu,Yw,t);                
                mf += f/B;

                xu = np.mean(self.X[:,v]+mu*self.mX[:,v],1);
                Yw = self.Y[:,w]+mu*self.mY[:,w];                

                dYw,dxu = dJ(xu,Yw,t);
                
                self.mX[:,v] = mu*self.mX[:,v]-dt*np.outer(dxu,np.ones(v.shape[0]))/(v.shape[0]*B);
                self.X[:,v] += self.mX[:,v];
                
                self.mY[:,w] = mu*self.mY[:,w]-dt*dYw;
                self.Y[:,w] += self.mY[:,w];
                
            print self.it,u,mf;
        self.it += 1;