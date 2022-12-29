import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile as wave
from collections import deque 

wavefile = "clean_guitar.wav"
SAMPLE_RATE, indata = wave.read(wavefile)
outdata = np.zeros(indata.shape)

class effect(object):
    def operate(self, v):
        return v

class simple_delay(effect):
    def __init__(self, delay_time, decay):
        global SAMPLE_RATE
        self.decay = decay
        nd = int((delay_time*SAMPLE_RATE)//1)
        self.buffer = deque(nd*[np.zeros(2)])
    def operate(self, v):
        self.buffer.append(v)
        return (1-self.decay)*v + self.decay*self.buffer.popleft()

class effect_chain(effect):
    def __init__(self, effects):
        self.effects = effects
    def add_effect(self, e):
        self.effects.append(e)
    def operate(self,v):
        vv = v
        for e in self.effects:
            vv = e.operate(vv)
        return vv

delay1 = simple_delay(0.25, .6)
delay2 = simple_delay(0.5, .2)
effects = effect_chain([delay1,delay2])

for i, a in enumerate(indata):
    outdata[i] = effects.operate(a)

wave.write("effect_chain_ex.wav", SAMPLE_RATE, outdata.astype(np.int16))