import numpy as np
import networkx as nx
from scipy.linalg import solve

def generate_hierarchical_network(N,mu,p0,recip=10):
    W = np.zeros((N,N))
    A = np.zeros((N,N))
    for n in range(0,N):
        deg = np.sum(A,1)
        rands = np.random.rand(n)
        for m in range(0,n):
            if (np.sum(deg)>0):
                p = deg[m]/np.sum(deg) + mu
            if np.sum(deg)==0:
                p=p0
            if rands[m]<p:
                weight = np.random.rand()
                W[n,m] = weight
                W[m,n] = weight/recip
                A[n,m] = 1
                A[m,n] = 1
    return(W)

def parameterise_network(W,eps):
  return((1-eps)*0.5*(W+np.transpose(W))+eps*W)

def henrici_departure(W):
  return(np.sqrt(np.linalg.norm(W,'fro')**2 - np.sum(np.abs(np.linalg.eig(W).eigenvalues)**2)))

def normalised_henrici_departure(W):
    return((np.sqrt(np.linalg.norm(W,'fro')**2 - np.sum(np.abs(np.linalg.eig(W).eigenvalues)**2)))/(np.sqrt(np.linalg.norm(W,'fro')**2)))

def weighted_reciprocity(W):
    N = W.shape[0]
    W_symm = np.zeros((N,N))
    Wcap = 0
    Wcap_symm = 0
    for i in range(0,N):
        for j in range(0,N):
            W_symm[i,j] = np.min([W[i,j],W[j,i]])
            if (i!=j):
                Wcap = Wcap+W[i,j]
                Wcap_symm = Wcap_symm+W_symm[i,j]
    return(Wcap_symm/Wcap)

def trophic_coherence(W):
    N = W.shape[0]
    colsum = np.sum(W,0)
    rowsum = np.sum(W,1)
    u = colsum + rowsum
    v = colsum - rowsum
    Lambda = np.diag(u) - W - np.transpose(W)
    G =  nx.from_numpy_array(W,create_using=nx.DiGraph)
    wcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    for l in range(0,len(wcc)):
        wc = sorted(wcc[l])
        Lambda[wc[0],wc[0]]=1
        v[wc[0]]=0
    h = solve(Lambda, v)
    for l in range(0,len(wcc)):
        wc = sorted(wcc[l])
        h[wc] = h[wc]-np.min(h[wc])*np.ones(len(wc))
    num =0
    denom =0
    for i in range(0,N):
        for j in range(0,N):
            num+=W[i,j]*(h[j]-h[i]-1)**2
            denom = denom + W[i,j]
    return([np.sqrt(1-num/denom),h])

def Laplacian(W):
    N = W.shape[0]
    colsum = np.sum(W,0)
    rowsum = np.sum(W,1)
    u = colsum + rowsum
    v = colsum - rowsum
    Lambda = np.diag(u) - W - np.transpose(W)
    return Lambda

def LaplaceEigenmodes(W):
    L = Laplacian(W)
    eigenValues, eigenVectors = np.linalg.eig(L)
    idx = eigenValues.argsort()
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return([eigenValues,eigenVectors])

def LaplaceEigenmodeProjection(W,X):
    eval, evec = LaplaceEigenmodes(W)
    Q = np.array(evec)
    return np.linalg.inv(Q)@X
