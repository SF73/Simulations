from numba import njit, prange
import numpy as np
from parameters import parameters
from stats import genLookupTable
import time


def simulate(param):
    if param.table.size == 0:
        param.table = genLookupTable(param.G,param.efficiency)
        
    #param.packetSize = max(int(300*param.eRate),int(np.round(1600/param.succes_proba)))
    attributes = {x:param.__getattribute__(x) for x in dir(param) if not x.startswith("_")}
    for key, value in attributes.items():
        print(f"{key}: {value}")
    results = run(param)
    return results
@njit(cache = True, parallel = True, nogil = True)
def run(param):
    Ne = int(param.T*param.eRate*1e9)
    hist = np.zeros(param.bins.size-1,dtype=np.int32)
    Nloop = int(np.round(Ne/param.packetSize))
    bins = param.bins
    N0_after_deadtime = 0
    N0_real = 0
    N1_after_deadtime = 0
    N1_real = 0
    p = param.succes_proba #proba d'avoir au moins 1 succes parmis G essais
    lookupTable = param.table
    for n in prange(Nloop):
        te = generateElectrons_fast(param.eRate, param.packetSize, p, param.T_On, param.T_Off)
        photons = generatePhotons_fast(param.tau, te, lookupTable)
        bunchSize = photons.size
        
        if bunchSize>0:
            if param.pulsedClock == True:
                maxRange = np.max(photons)
                clock = np.arange(0,maxRange,(param.T_On+param.T_Off))
                clock += param.clockDelay
                
                N1_real += bunchSize
                #clock = applyDeadTime(clock,deadTime)
                clock = norm(0,param.jitter,clock)
                photons += param.detectorDelay
                photons = applyDeadTime(photons,param.td)
                N1_after_deadtime += len(photons)
                photons = norm(0,param.jitter,photons)
                
                delays = correlate(clock,photons)
                hist += np.histogram(delays,bins)[0]
            else:
                a=np.random.rand(bunchSize)<param.BS
                S = a.sum()
                if ((S>0) and ((bunchSize - S) > 0)):
                    
                    clock = photons[~a] + param.clockDelay
                    
                    detector = photons[a] + param.detectorDelay
                    
                    N0_real += clock.size
                    N1_real += detector.size
                    clockMask = np.random.rand(clock.size)<param.clockFilter
                    detectorMask = np.random.rand(detector.size)<param.detectorFilter
                    if ((detectorMask.sum()>0) and (clockMask.sum()>0)):
                        clock = clock[clockMask]
                        detector = detector[detectorMask]
                        
                        clock = norm(0,param.jitter,clock)
                        detector = norm(0,param.jitter,detector)
                        clock = applyDeadTime(clock,param.td)
                        detector = applyDeadTime(detector,param.td)
                        N0_after_deadtime += clock.size
                        N1_after_deadtime += detector.size
                        delays = correlate(clock,detector)
                        
                        hist += np.histogram(delays,bins)[0]
    return N0_real, N0_after_deadtime, N1_real, N1_after_deadtime, hist


@njit(cache=True,nogil=True)
def applyDeadTime(array,deadTime):
    if len(array)<2:
        return array
    i=0
    c=0
    onTime=np.empty_like(array)
    onTime[0]=array[0]
    while i < (len(array)-1):
        for j in range(i+1,len(array)):
            #pour chaque element on regarde quand le dead time est fini
            if (array[j]>(array[i]+deadTime)):
                #delai plus grand que deadTime
                #on garde le delai
                c = c+1
                onTime[c]=array[j]
                #on saute les photons compris dans le deadTime
                i=j
                break
        if (j==len(array)-1):
            break
    return onTime[0:c+1]

@njit(cache=True,nogil=True)
def correlate(clock,detector):
    #start multistop
    delays = []
    j=len(detector)-1
    for i in range(len(clock)-1,-1,-1):
        #On parcourt les clocks antichronologiquement pour trouver la dernière
        while detector[j]-clock[i]>=0:
            #On regarde les photons du detecteur antichronologiquement
            #On ajoute les différences de temps jusqu'à trouver un photon detecteur
            #anterieur à la clock actuelle
            delays.append(detector[j]-clock[i])
            j-=1
            if j<0:break
        #detecteur anterieur à la clock => on passe a la clock précédente
        if j<0:break
    return np.array(delays)

@njit(cache=True,nogil=True)
def correlate_ss(clock,detector):
    #start stop
    delays = []
    j=len(detector)-1
    for i in range(len(clock)-1,-1,-1):
        #On parcourt les clocks antichronologiquement pour trouver la dernière
        while detector[j]-clock[i]>=0:
            #On regarde les photons du detecteur antichronologiquement
            #On ajoute les différences de temps jusqu'à trouver un photon detecteur
            #anterieur à la clock actuelle
            j-=1
            if j<0:break
        delays.append(detector[j+1]-clock[i])
        #detecteur anterieur à la clock => on passe a la clock précédente
        if j<0:break
    return np.array(delays)

@njit(cache=True,nogil=True)
def norm(loc,sig,arr):
    """
    Add jitter.
    Warning : Inplace modification !

    Parameters
    ----------
    loc : float
        Mean.
    sig : float
        standard deviation.
    arr : np.float[:]
        Arrival times.

    Returns
    -------
    arr : TYPE
        DESCRIPTION.

    """
    if loc == 0 and sig == 0:
        return arr
    for i in range(arr.size):
        arr[i] += np.random.normal(loc,sig)
    return arr

@njit(cache=True)
def TruncBin(lookupTable,size):
    return np.searchsorted(lookupTable,np.random.rand(int(size)))

#float64[:](float64,int32,float32,float32,float32),
@njit(cache = True,nogil=True)
def generateElectrons_fast(electronRate,packetSize,p,T_On,T_Off):
    #Rate correction to generate only electrons with one successful photon
    newRate = electronRate * p
    
    #Number of electron to generate
    Ne = np.random.binomial(packetSize,p)
    
    #Absolute arrival time of poissonnian electron is given by cumsum of \
    #exponential distribution
    te = (-1/newRate*np.log1p(-np.random.rand(Ne))).cumsum()
    
    #if the beamblanker is used, only electron during T_On state can reach the sample
    # if T_Off > 0 and T_On > 0:
    #     #te%(T_On+T_Off) match each electron to a period
    #     # we keep the electron only if is time inside the period is smaller than T_On
    #     #assuming that a period is T_On ns of on-state then T_Off ns of off-state
    #     te = te[(te%(T_On+T_Off))<T_On]
    return te

#float64[:](float32,float64[:],float64[:]),
@njit(cache = True, nogil=True)
def generatePhotons_fast(tau, te, lookupTable):
    photonBunch = np.repeat(te,TruncBin(lookupTable,te.size))
    photonBunch += (-tau*np.log1p(-np.random.rand(photonBunch.size)))
    photonBunch.sort()
    return photonBunch

# @njit(cache=True,nogil=True)
# def generateBunches2(electronRate,PacketSize,tau,G,efficiency,lookupTable):  
#     '''
#     Use the truncated binomial law to generate only the photons
#     that can reach one detector

#     Parameters
#     ----------
#     electronRate : TYPE
#         DESCRIPTION.
#     PacketSize : TYPE
#         DESCRIPTION.
#     tau : TYPE
#         DESCRIPTION.
#     G : TYPE
#         DESCRIPTION.
#     efficiency : TYPE
#         DESCRIPTION.
#     lookupTable : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     photonBunch : TYPE
#         DESCRIPTION.

#     '''
#     p = 1-(1-efficiency)**G #proba d'avoir au moins 1 succes parmis G essais
#                             #proba de succes = efficiency
    
#     eRate = electronRate * p#on ajuste la frequence moyenne des electrons
    
#     Ne = np.random.binomial(PacketSize,p)
#     Ng = TruncBin(lookupTable, Ne)
    
#     te = (-1/eRate*np.log1p(-np.random.rand(Ne))).cumsum()
    
#     #nombre de photons par electron en tenant compte de l efficacite
#     # Garray = TruncBin(lookupTable,te.size)
#     photonBunch = np.repeat(te,Ng)
    

#     #genere les temps entre les photon d'un bunch
#     #distrib exponentielle exp(-tau)    
#     photonBunch += (-tau*np.log1p(-np.random.rand(photonBunch.size)))
#     photonBunch.sort()
    
#     #np.random.exponential(tau,te.size)
#     #photonBunch.sort()
#     # photonBunch = te+np.random.exponential(tau,te.size)
    
#     return photonBunch

def test():
    param = parameters()
    param.I = 40
    param.T = 10
    param.BS = 0.5
    param.G = 10
    param.dt = 64
    param.efficiency = 1e-2
    param.td = 86
    param.tau = 8
    param.detectorDelay = 11
    param.clockDelay = 0
    start = time.time()
    results = simulate(param)
    stop = time.time()
    print(stop-start)
    
    return results, param