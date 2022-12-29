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

delay1 = simple_delay(0.25, .6)
delay2 = simple_delay(0.5, .2)

for i, a in enumerate(indata):
    outdata[i] = delay2.operate(delay1.operate(a))

wave.write("delay_plugin_ex.wav", SAMPLE_RATE, outdata.astype(np.int16))