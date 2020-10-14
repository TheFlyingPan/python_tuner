#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from scipy.io.wavfile import write, read
import scipy.io.wavfile as wavfile
import scipy
from scipy.fftpack import fft
import numpy as np
from matplotlib import pyplot as plt
import sounddevice as sd

%matplotlib tk

fs_rate, signal = wavfile.read("output.wav")
print("Frequency sampling", fs_rate)
l_audio = len(signal.shape)
print("Channels", l_audio)
if l_audio == 2:
    signal = signal.sum(axis=1) / 2
N = signal.shape[0]
print("Complete Samplings N", N)
secs = N / float(fs_rate)
print("secs", secs)
Ts = 1.0/fs_rate  # sampling interval in time
print("Timestep between samples Ts", Ts)
t = np.arange(0, secs, Ts)  # time vector as scipy arange field / numpy.ndarray
FFT = abs(fft(signal))
FFT_side = FFT[range(N//2)]  # one side FFT range
freqs_side = freqs[range(N//2)]  # one side frequency range
index = np.where(FFT_side == np.amax(FFT_side))
print(freqs_side[index[0]])
# plotting the positive fft spectrum
p3 = plt.plot(freqs_side, abs(FFT_side), "b")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count single-sided')
plt.show()


def record3s():
    fs = 44100
    seconds = 3
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write('output.wav', fs, myrecording)
