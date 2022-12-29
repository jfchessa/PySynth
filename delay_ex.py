import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile as wave

wavefile = "clean_guitar.wav"
samplerate, data = wave.read(wavefile)
plt.plot(data)
plt.show()

# add a simple slap-back delay
delay = 0.3  # seconds
decay = 0.25

nd = int((delay*samplerate)//1)
ddata = (1-decay)*data
ddata[nd:] += decay*data[:-nd]
wave.write("delay_example.wav", samplerate, ddata.astype(np.int16))