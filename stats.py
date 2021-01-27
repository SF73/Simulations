import numpy as np
from decimal import Decimal
def Cnk(n,k):
    return np.math.factorial(n)//(np.math.factorial(k)*np.math.factorial(n-k))

def posBin(k,n,p):
    return Cnk(n,k)*np.power(p,k)*np.power((1-p),(n-k))/(1-np.power((1-p),n))

def genLookupTable(n,p):
    #on genere la table de proba de la loi binomial positive
    #ce qui correspond a la proba d'avoir k reussite parmis n si la proba de
    #reussite est p et dans le cas ou k>=1
    lookupTable = [posBin(int(k),int(n),p) for k in np.arange(1,n+1,dtype=np.int)]
    # lookupTable = np.array([0]+lookupTable)
    #cdf
    lookupTable = np.cumsum([0.]+lookupTable)
    lookupTable /= lookupTable[-1]
    idx = np.where(lookupTable<1)[0].max()
    lookupTable=lookupTable[0:idx+2]
    return lookupTable

def multinomial_prob(G,n1,n2,p):
    """
    Parameters
    ----------
    G : int
        Number of photon per bunch
    n1 : int
        Number of photon reaching D1.
    n2 : int
        Number of photon reaching D2.
    p : list of float
        [pvoid,p1,p2] with pvoid proba of not reaching D1 or D2, p1 (resp. p2)\
            proba of reaching D1 (resp. D2).

    Returns
    -------
    float
        probability of the given state in the model.

    """
    psum = np.sum(p)
    if abs(1-psum) > 1e-8:
        print(f"Warning sum of proba not equal to 1 : psum = {psum}")
    if n1+n2 > G:
        raise ValueError("n1 + n2 cannot be greater than G")
    n0 = G-n1-n2
    C = np.math.factorial(G)/(np.math.factorial(n1)*np.math.factorial(n2)*np.math.factorial(n0))
    return C*np.power(p[0],n0)*np.power(p[1],n1)*np.power(p[2],n2)

def generate_all_proba(G,p):
    """
    Return all the possible states for a given G with the corresponding proba
    
    Parameters
    ----------
    G : int
        Number of photon per bunch
    p : list of float
        [pvoid,p1,p2] with pvoid proba of not reaching D1 or D2, p1 (resp. p2)\
            proba of reaching D1 (resp. D2).

    Returns
    -------
    states : np.array(n1,n2,nvoid)
        All the possibles configurations for n1+n2+nvoid = G
    probas : np.array(proba)
        probas[i] : probability of the state states[i]

    """
    states = []
    probas = []
    for n1 in range(G):
        for n2 in range(G):
            if n1 + n2 > G:break
            probas.append(multinomial_prob(G,n1,n2,p))
            states.append([n1,n2,G-(n1+n2)])
    states = np.array(states)
    probas = np.array(probas)
    return states, probas

def list_proba(G, BS, efficiency):
    p = [1-efficiency,efficiency*BS,efficiency*(1-BS)]
    states, probas = generate_all_proba(G,p)
    n1, n2, nvoid = states[:,0], states[:,1], states[:,2]
    correlation_mask = np.bitwise_and(n1>0,n2>0)
    pileup_mask = np.bitwise_or(np.bitwise_and(n1>=1,n2>1), np.bitwise_and(n2>=1,n1>1))
    d1_mask = n1>0
    d2_mask = n2>0
    
    p_correlation = probas[correlation_mask].sum()
    p_pileup = probas[pileup_mask].sum()
    p_d1 = probas[d1_mask].sum()
    p_d2 = probas[d2_mask].sum()
    return {"p_d1" : p_d1, 
     "p_d2" : p_d2, 
     "p_correlation" : p_correlation, 
     "p_pileup" : p_pileup}

def proba_analytique(G, BS, efficiency):
    p1 = efficiency * BS #detecteur
    p2 = efficiency * (1-BS) #clock
    p0 = (1-efficiency) #void
    
    p_d1 = 1-(1-p1)**G
    p_d2 = 1-(1-p2)**G
    p_correlation = p_d1 + p_d2 + p0**G -1
    p_pileup = p_correlation - G*(G-1)*p0**(G-2)*p1*p2
    
    return {"p_d1" : p_d1, 
     "p_d2" : p_d2, 
     "p_correlation" : p_correlation, 
     "p_pileup" : p_pileup}
    # print(f"p_correlation : {p_correlation}")
    # print(f"p_pileup : {p_pileup}")
    # print(f"p_pileup/p_correlation : {p_pileup/p_correlation}")
    # print(f"p_d1 : {p_d1}")
    # print(f"p_d2 : {p_d2}")
    
