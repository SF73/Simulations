import numpy as np


def Cnk(n,k):
    return np.math.factorial(n)//(np.math.factorial(k)*np.math.factorial(n-k))

def posBin(k,n,p):
    return Cnk(n,k)*np.power(p,k)*np.power((1-p),(n-k))/(1-np.power((1-p),n))

def genLookupTable(n,p):
    #on genere la table de proba de la loi binomial positive
    #ce qui correspond a la proba d'avoir k reussite parmis n si la proba de
    #reussite est p et dans le cas ou k>=1
    lookupTable = [posBin(k,n,p)for k in np.arange(1,n+1,dtype=np.float64)]
    # lookupTable = np.array([0]+lookupTable)
    #cdf
    lookupTable = np.cumsum([0.]+lookupTable)
    lookupTable /= lookupTable[-1]
    idx = np.where(lookupTable<1)[0].max()
    lookupTable=lookupTable[0:idx+2]
    return lookupTable