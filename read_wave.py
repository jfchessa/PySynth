import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile as wave

wavefile = "clean_guitar.wav"
samplerate, data = wave.read(wavefile)
plt.plot(data)
