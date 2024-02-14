import numpy as np
import control as ct
from scipy.linalg import eig

def network_ornstein_uhlenbeck(W,sigma,Theta,gamma,dt,steps,X0=None):
    N = W.shape[0]
    if X0 is None:
        X0 = np.zeros(N)
    I = np.diag(np.ones(N))
    t = np.linspace(0,dt*steps,steps+1)
    X = np.zeros((N,steps+1))
    X[:,0] = X0
    for idx in range(1,steps+1):
        X[:,idx] = X[:,idx-1] + dt*(Theta*(gamma*W-I)@X[:,idx-1]) + np.random.multivariate_normal(np.zeros(N),(np.sqrt(2*sigma)*I*dt))
    return [X,t]


def EPR_network_ornstein_uhlenbeck(W,sigma,Theta,gamma):
    N = W.shape[0]
    I = np.diag(np.ones(N))
    B = Theta*(I-gamma*W)
    S = ct.lyap(B,-2*sigma*I)
    Q = B@S-sigma*I
    Phi = -np.trace((1/sigma)*B@Q)
    return(Phi)

def steady_state(TP):
    w, vl, vr = eig(TP, left=True)
    perron = vl[:,np.argmax(np.abs(w))]
    return np.abs(perron)

def EPR_markov_chain(ss,TP):
    N = len(ss)
    Phi=0
    for i in range(0,N):
        for j in range(0,N):
            Phi+= TP[i,j]*ss[j]*np.log((TP[i,j]*ss[j])/(TP[j,i]*ss[i]))
    return Phi

def markov_chain_to_JTP(MC,N):
    JTP = np.zeros((N,N))
    L = len(MC)
    for t in range(1,L):
        JTP[MC[t-1],MC[t]]+=1
    return JTP/(L-1)


def EPR_markov_chain_joint(JTP):
    N = JTP.shape[0]
    Phi=0
    for i in range(0,N):
        for j in range(0,N):
            if np.min([JTP[i,j],JTP[j,i]])>0:
                Phi+= JTP[i,j]*np.log((JTP[i,j])/(JTP[j,i]))
            elif np.max([JTP[i,j],JTP[j,i]])>0:
                print('Error: EPR diverges')
                return None
    return Phi

def EPR_markov_chain_joint_JSD(JTP):
    N = JTP.shape[0]
    Phi=0
    Mix = 0.5*(JTP + np.transpose(JTP))
    for i in range(0,N):
        for j in range(0,N):
            if np.min([JTP[i,j],JTP[j,i]])>0:
                Phi+= 0.5*JTP[i,j]*np.log((JTP[i,j])/(Mix[i,j])) + 0.5*JTP[j,i]*np.log((JTP[j,i])/(Mix[i,j]))
    return Phi