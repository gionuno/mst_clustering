#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 22:58:33 2017

@author: quien
"""

import numpy as np;
import numpy.random as rd;
import matplotlib.pyplot as plt;

def prim(A):
    C = np.inf*np.ones(A.shape[0]);
    E = -np.ones(A.shape[0],dtype=int);
    
    F = [];
    
    Q = range(A.shape[0]);
    while len(Q)>0:
        j = np.argmin(C[Q]);
        u = Q[j];
             
        del Q[j];
        if E[u] >= 0:
            F = F + ([[u,E[u]]] if u < E[u] else [[E[u],u]]);
        for v in range(A.shape[0]):
            if u != v and (v in Q) and A[u,v] < C[v]:
                C[v] = A[u,v];
                E[v] = u;
    return np.array(F);

def find(P,x):
    if P[x] != x:
        P[x] = find(P,P[x]);
    return P[x];

def join(P,D,x,y):
    x_p = find(P,x);
    y_p = find(P,y);
    if x_p == y_p:
        return;
    if D[x_p] < D[y_p]:
        P[x_p] = y_p;
    elif D[x_p] > D[y_p]:
        P[y_p] = x_p;
    else:
        P[y_p] = x_p;
        D[x_p] +=  1;

def hemst(X,k):
    Y = [np.copy(X)];
    
    P = [];
    D = [];
    C = [];
        
    t = 0;
    n = 1;
         
    while n != k:
        A = np.zeros((Y[t].shape[0],Y[t].shape[0]));
    
        P.append(np.arange(Y[t].shape[0],dtype=int));

        D.append( np.zeros(Y[t].shape[0],dtype=int));
        C.append( np.zeros(Y[t].shape[0],dtype=int));
                
        for i in range(A.shape[0]):
            for j in range(i+1,A.shape[0]):
                A[i,j] = A[j,i] = np.linalg.norm(Y[t][i,:]-Y[t][j,:]);
        
        m = 0.0;
        for i in range(A.shape[0]):
            for j in range(i+1,A.shape[0]):
                m += A[i,j];
        m /= 0.5*A.shape[0]*(A.shape[0]-1);
    
        s = 0.0;
        for i in range(A.shape[0]):
            for j in range(i+1,A.shape[0]):
                s += (A[i,j]-m)**2;
        s /= 0.5*A.shape[0]*(A.shape[0]-1);
        s = np.sqrt(s);
        
        T = prim(A);
        n_T = [];
        for [u,v] in T:
            if A[u,v] > m + s:
                n += 1;
            else:
                n_T += [[u,v]];
        T = n_T;
        
        if n < k:
            aux = np.argsort([-A[T[i][0],T[i][1]] for i in range(len(T))]);
            for idx in aux[k-n:]:
                join(P[t],D[t],T[idx][0],T[idx][1]);
            c = 0;
            for i in range(A.shape[0]):
                if find(P[t],i) == i:
                    C[t][i] = c;
                    c += 1;
            for i in range(A.shape[0]):
                C[t][i] = C[t][find(P[t],i)];
            return Y,C;
        for [u,v] in T:
            join(P[t],D[t],u,v);
        c = 0;
        for i in range(A.shape[0]):
            if find(P[t],i) == i:
                C[t][i] = c;
                c += 1;
        for i in range(A.shape[0]):
            C[t][i] = C[t][find(P[t],i)];
        if n > k:
            Z = np.zeros((n,Y[t].shape[1]));
            N = np.zeros(n);
            for i in range(A.shape[0]):
                Z[C[t][i],:] += Y[t][i,:];
                N[C[t][i]]   += 1.0;
            for j in range(n):
                Z[j,:] /= N[j];
            D = np.zeros(n);
            I = -np.ones(n,dtype=int);
            for i in range(A.shape[0]):
                d = np.linalg.norm(Z[C[t][i],:]-Y[t][i,:]);
                if I[C[t][i]] < 0 or d < D[C[t][i]]:
                    I[C[t][i]] = i;
                    D[C[t][i]] = d;
            Y.append(Y[t][I,:]);
        t += 1;
    return Y,C;


X = np.r_[2e-1*rd.randn(100,2)+1.0,
          2e-1*rd.randn(100,2),
          2e-1*rd.randn(100,2)-1.0,
          2e-1*rd.randn(100,2)+np.array(100*[[-1.0,1.0]]),
          2e-1*rd.randn(100,2)+np.array(100*[[1.0,-1.0]])];

Y,C = hemst(X,5);
          
plt.scatter(X[:,0],X[:,1],c=C[0]);   

Z = np.loadtxt('ecoli.csv',delimiter=' ');

C = np.asarray(Z[:, -1],dtype=int);
X = Z[:,:-1];
X_c = np.dot(X,np.eye(X.shape[1]) - np.ones((X.shape[1],X.shape[1]))/X.shape[1]);
X_c /= np.outer(np.linalg.norm(X_c,axis=1),np.ones((X.shape[1],1)))+1e-5;
X_c -= np.outer(np.ones((X.shape[0],1)),np.mean(X_c,axis=0));
X_c = np.dot(X_c,np.linalg.pinv(np.dot(X_c.T,X_c)));
                     
Y,D = hemst(X_c,4);
          
plt.scatter(X_c[:,0],X_c[:,1],c=C);

plt.scatter(X_c[:,0],X_c[:,1],c=D[0]);

