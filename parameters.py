import numba as nb
from numba.experimental import jitclass
import numpy as np
from .stats import genLookupTable

spec = [
('I', nb.float32),
('G', nb.int32),
('tau', nb.float32),
('T', nb.float32),
('paralyzableDeadTime', nb.boolean),
('td', nb.float32),
('BS', nb.float32),
('jitter', nb.float32),
('clockFilter', nb.float32),
('detectorFilter', nb.float32),
('clockDelay', nb.float32),
('detectorDelay', nb.float32),
('efficiency', nb.float64),
('dt', nb.int32),
('binNumber', nb.int32),
('T_On', nb.float32),
('T_Off', nb.float32),
('packetSize', nb.int32),
('pulsedClock', nb.boolean),
('table',nb.float64[:])
]

@jitclass(spec)
class parameters(object):
    '''Class to hold all the parameters
    
    Parameters
    ----------
    I : float
        Ebeam current in pA
    G : int
        number of electron-hole pair per electron
    tau : float
        lifetime in ns
    T : float
        Integration time in s
    paralyzableDeadTime : bool
        Whether the deadtime is paralyzable or not (default: False)
    td : float
        dead-time of the TCSPC in ns
    BS : float
        Ndetector/(Ndetector+Nclock)
    jitter : float
        jitter on each channel of the TCSPC (standard deviation of the normal)
    clockFilter : float
        filter only on the clock
    detectorFilter : float
        filter only on the detector
    clockDelay : float
        Delay added on channel0 in ns
    detectorDelay : float
        Delay added on channel1 in ns
    efficiency : float
        photon collected/electron-hole pair generated
    binNumber : int
        number of bin
    dt : int
        bin\'s width in ps
    T_On : float
        Duration of ON time of the beamblanker
    T_Off : float
        Duration of OFF time of the beamblanker
    packetSize : int
        number of electron to generate per loop
    table : float64[:]
        Lookup table for truncated binomial law
    pulsedClock : bool
        Whether the clock a detector or a pulse generator (default: False)
    '''
    def __init__(self):
        self.clockFilter = 1
        self.detectorFilter = 1
        self.jitter = 0
        self.packetSize = 10000
        self.dt = 4
        self.T_On = 0
        self.T_Off = 0
        self.binNumber = int(2**16)
        self.pulsedClock = False
        self.paralyzableDeadTime = False
    @property
    def bins(self):
        return np.arange(0,self.binNumber*self.dt*1e-3,self.dt*1e-3)
    @property
    def eRate(self):
        return (self.I/1.602176634e-19)*1e-21
    
    '''
    succes_proba : float
        proba d'avoir au moins 1 succes parmis G essais
        proba de succes = efficiency
    '''
    @property
    def succes_proba(self):
        return 1-(1-self.efficiency)**self.G