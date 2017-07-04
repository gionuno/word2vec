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

def J(x,w,t):
    s = sigm(t*np.dot(w,x));
    return -np.sum(np.log(s));

def dJ(x,w,t):
    s  = sigm(t*np.dot(w,x));
    dw = -np.outer(t*(1.0-s),x);
    dx = -np.dot(t*(1.0-s),w);
    return dw,dx;

class word2vec:
    def __init__(self,U,G,D):
        self.U = U; #dict of lists of 2K words
        self.G = G; #dict of adyacency graph for sampling
        self.D = D; #dimen
        self.V = len(G);
        
        self.a  = 0.5*rd.randn(self.V,self.D)/self.V;
        self.da = np.zeros(self.a.shape);
        self.ma = np.zeros(self.a.shape);
        
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
                aw = self.a[w,:]+dt*mu*self.ma[w,:];
                
                f = J(xu,aw,t);
                print u, f;
                
                daw,dxu = dJ(xu,aw,t);
                
                self.ma[w,:] = mu*self.ma[w,:]-dt*daw;
                self.mX[:,v] = mu*self.mX[:,v]-dt*np.outer(dxu,np.ones(len(v)));
            
                self.a += self.ma;
                self.X += self.mX;
            