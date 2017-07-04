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
    return -np.sum(np.log(s));

def dJ(x,Y,t):
    s  = sigm(t*np.dot(Y.T,x));
    dw = -np.outer(t*(1.0-s),x).T;
    dx = -np.dot(t*(1.0-s),Y.T);
    return dw,dx;

class word2vec:
    def __init__(self,U,G,D):
        self.U = U; #dict of lists of 2K words
        self.G = G; #dict of adyacency graph for sampling
        self.D = D; #dimen
        self.V = len(G);
        
        self.X  = rd.randn(self.D,self.V);
        self.dX = np.zeros(self.X.shape);
        self.mX = np.zeros(self.X.shape);
        
        self.IT = 10;
        self.it = 0;
    
    def get_tup(self,u):
        r = rd.randint(len(self.U[u]));
        return self.U[u][r];
    
    def get_wor(self,u,K):
        w = [u];
        t = [1.0];
        for k in range(K):
            r = rd.randint(self.V);
            while r in self.G[u]:
                r = rd.randint(self.V);
            w.append(r);
            t.append(-1.0);
        return np.array(t),np.array(w);
    
    def step(self,B,K,dt,mu):
        for u in range(self.V):
            for b in range(B):
                v   = self.get_tup(u);
                t,w = self.get_wor(u,K);
            
                mv = self.mX[:,v];
                xv = self.X[:,v];
            
                xu = np.sum(xv+dt*mu*mv,axis=1);
                Yw = self.X[:,w]+dt*mu*self.mX[:,w];
                
                f = J(xu,Yw,t);
                print u, f;
                
                dYw,dxu = dJ(xu,Yw,t);
                mmX = np.zeros(self.mX.shape);
                mmX[:,w] += dYw;
                mmX[:,v] += np.outer(dxu,np.ones(len(v)));
                
                self.mX = mu*self.mX-dt*mmX;
                self.X += self.mX;
            