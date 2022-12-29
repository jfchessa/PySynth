import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile as wave
from collections import deque 

wavefile = "clean_guitar.wav"
SAMPLE_RATE, indata = wave.read(wavefile)
outdata = np.zeros(indata.shape)

def plot_signal(data):
    """
    Plots the left and right channel wave forms
    """
    f, (right, left) = plt.subplots(2, 1, sharey=True)
    right.plot(data[:,0])
    left.plot(data[:,1])
    plt.show()

class effect(object):
    """
    Just a base effect
    """
    def __init__(self, nchan=2):
        self.nchan = nchan
    def operate(self, v):
        return v

class effect_chain(effect):
    """
    A class that holds a list of effects.  The operate() command acts as if
    all the effects were in series
    """
    def __init__(self, effects):
        self.effects = effects
    def add_effect(self, e):
        self.effects.append(e)
    def __repr__(self):
        s = "->"
        for e in self.effects:
            s += str(e)+"->"
    def operate(self,v):
        vv = v
        for e in self.effects:
            vv = e.operate(vv)
        return vv

class sine_oscil(effect):
    """
    Sine wave ocsillator
    """
    def __init__(self, freq, amp=1.0, phase=0.0, nchan=1):
        effect.__init__(self, nchan)
        global SAMPLE_RATE
        self.wt = 2*np.pi*freq/SAMPLE_RATE
        self.amp = amp
        self.phi = phase*(np.pi/180)
        self.count = 0
    def __repr__(self):
        return "Sine Oscillator"
    def eval(self):
        v = self.amp*np.sin(self.wt*self.count + self.phi)
        self.count += 1
        return v*np.ones(self.nchan)

class adsr_envelope(effect):
    """
    Attack, decay, sustain, release envelope
    att = attack duration
    dec = decay duration
    slev = sustain level
    rel = release duration

         ^
        /  \_____
       /          \
    """ 
    def __init__(self, att, dec, slev, rel, nchan=1):
        effect.__init__(self, nchan)
        global SAMPLE_RATE
        dt = 1/SAMPLE_RATE
        self.att = att*SAMPLE_RATE
        self.dec = dec*SAMPLE_RATE
        self.slev = slev
        self.rel = rel*SAMPLE_RATE
        self.count = 0
    def __repr__(self):
        return "ADSR Envelope"
    def eval(self, duration=np.inf):
        global SAMPLE_RATE
        dur = duration*SAMPLE_RATE
        if self.count <= self.att:             # attack
            v = self.count/self.att
        elif self.count <= self.att+self.dec:  # decay
            v = 1.0 - (self.count-self.att)*(1.0-self.slev)/(self.dec)
        elif self.count <= dur - self.rel:     # sustained
            v = self.slev
        elif self.count >= dur:                # released
            v = 0.0
        else:                                  # release
            v = (dur - self.count)*(self.slev/self.rel)
        self.count += 1
        return v*np.ones(self.nchan)

class clean_boost(effect):
    def __init__(self, boost, nchan=2):
        effect.__init__(self, nchan)
        self.boost = boost
    def __repr__(self):
        return "Clean Boost"
    def operate(self, v):
        return self.boost*v

class low_pass_filter(effect):
    def __init__(self, cutoff_freq, boost=1.0, nchan=2):
        effect.__init__(self, nchan)
        global SAMPLE_RATE
        dt = 1/SAMPLE_RATE
        c1 = 2*np.pi*dt*cutoff_freq
        self.alpha = c1/(c1+1.0)
        self.boost = boost
        self.hist = np.zeros(self.nchan)
    def __repr__(self):
        return "1st Order Low Pass Filter"
    def operate(self, v):
        vv = self.alpha*v + (1-self.alpha)*self.hist
        self.hist = vv
        return self.boost*vv

class high_pass_filter(effect):
    def __init__(self, cutoff_freq, boost=1.0, nchan=2):
        effect.__init__(self, nchan)
        global SAMPLE_RATE
        dt = 1/SAMPLE_RATE
        c1 = 2*np.pi*dt*cutoff_freq
        self.alpha = 1.0/(c1+1.0)
        self.boost = boost
        self.histv = np.zeros(self.nchan)
        self.histvv = np.zeros(self.nchan)
    def __repr__(self):
        return "1st Order High Pass Filter"
    def operate(self, v):
        vv = self.alpha*(self.histvv + v - self.histv)
        self.histvv = vv
        self.histv = v
        return self.boost*vv

class bandpass_filter(effect):
    def __init__(self, center_freq, bandwidth, boost=1.0, nchan=2):
        effect.__init__(self, nchan)
        self.hpf = high_pass_filter(center_freq - bandwidth/2)
        self.lpf = low_pass_filter(center_freq + bandwidth/2)
        self.boost = boost
    def __repr__(self):
        return "1st Order Bandpass Filter"
    def operate(self, v): 
        vv = self.hpf.operate(v)
        vv = self.lpf.operate(vv)
        return self.boost*vv

class notch_filter(effect):
    def __init__(self, center_freq, bandwidth, boost=1.0, nchan=2):
        effect.__init__(self, nchan)
        self.lpf = high_pass_filter(center_freq - bandwidth/2)
        self.hpf = low_pass_filter(center_freq + bandwidth/2)
        self.boost = boost
    def __repr__(self):
        return "1st Order Notch Filter"
    def operate(self, v): 
        vl = self.lpf.operate(v)
        vh = self.hpf.operate(v)
        return self.boost*0.5*(vl+vh)

class auto_wah(effect):
    def __init__(self, rate, f1, f2, nchan=2):
        effect.__init__(self, nchan)
        self.foscil = sine_oscil(rate, abs(f2-f1)/2)
        self.fdc = (f1+f2)/2
        self.hist = np.zeros(self.nchan)
    def __repr__(self):
        return "Auto-Wah (disabled)"
    def operate(self, v):
        global SAMPLE_RATE
        dt = 1/SAMPLE_RATE
        fc = self.foscil.eval() + self.fdc
        c1 = 2*np.pi*dt*fc
        alpha = c1/(c1+1.0)
        vv = alpha*v + (1-alpha)*self.hist
        self.hist = vv
        return vv

class simple_delay(effect):
    def __init__(self, delay_time, decay, nchan=2):
        effect.__init__(self, nchan)
        global SAMPLE_RATE
        self.decay = decay
        nd = int((delay_time*SAMPLE_RATE)//1)
        self.buffer = deque(nd*[np.zeros(self.nchan)])
    def __repr__(self):
        return "Simple Delay"
    def operate(self, v):
        self.buffer.append(v)
        return (1-self.decay)*v + self.decay*self.buffer.popleft()

class digital_delay(effect):
    def __init__(self, delay_time, decay_rate, mix, nchan=2):
        effect.__init__(self, nchan)
        global SAMPLE_RATE
        nd = int((delay_time*SAMPLE_RATE)//1)
        self.decay_factor = decay_rate**(1/nd)
        self.mix = mix
        self.buffer = deque(nd*[np.zeros(self.nchan)])
    def __repr__(self):
        return "Digital Delay"
    def operate(self, v):
        vv = (1-self.mix)*v + self.mix*self.buffer.popleft()
        self.buffer.append(self.decay_factor*vv)
        return vv

class hard_clip(effect):
    def __init__(self, clip_level, nchan=2):
        effect.__init__(self, nchan)
        self.clip_level = clip_level
    def __repr__(self):
        return "Hard Clipping Distortion"
    def operate(self, v):
        vv = v
        for i, vvi in enumerate(vv):
            if abs(vvi) >= self.clip_level:
                vv[i] = np.sign(vvi)*self.clip_level
        return vv

class gr300(effect):
    def __init__(self, nchan=2):
        effect.__init__(self, nchan)
        self.lpf = low_pass_filter(100)
        self.rise = 0.1*np.ones(self.nchan)
        self.count = 0
        self.last = 0.0
    def __repr__(self):
        return "Roland GR 300 Emulator"
    def operate(self, v):
        vf = self.lpf.operate(v)
        if self.count > 0 and self.last*vf[0] < 0: # zero crossing
            self.count = 0
        vv = self.rise*self.count
        self.count += 1
        self.last = vf[0]
        return vv

#---------------------------------------------------------------------
delay = digital_delay(0.5, 0.5, 0.5)
dist = hard_clip(2000)
boost = clean_boost(4.0)
lpf = low_pass_filter(100)
hpf = high_pass_filter(500, 1.75)
notch = notch_filter(150, 100, 1.5)
wah = auto_wah(2, 100, 5000)
#effects = effect_chain([dist,boost,delay])
effects = effect_chain([wah, dist, delay, boost])
synth = gr300()

gen = sine_oscil(220, 4000, nchan=2)
env = adsr_envelope(.1,.3,.5,.2)
instdata = np.zeros(indata.shape)
#---------------------------------------------------------------------
for i, a in enumerate(indata):
    outdata[i] = synth.operate(a)
    #outdata[i] = effects.operate(a) 
    #instdata[i] = env.eval(3.0)*gen.eval()
    #instdata[i] = dist.operate(env.eval(3.0)*gen.eval())

wave.write("effects.wav", SAMPLE_RATE, outdata.astype(np.int16))
#wave.write("oscillator.wav", SAMPLE_RATE, instdata.astype(np.int16))
plot_signal(outdata)