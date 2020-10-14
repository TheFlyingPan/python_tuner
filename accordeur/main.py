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
from my_lib import convert
import collections

def record3s():
    fs = 44100
    seconds = 3
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write('output.wav', fs, myrecording)

record3s()
strings = collections.OrderedDict([
    ("Mi grave, E2", 82.41),
    ("La, A2", 110.00),
    ("Re, D3", 146.83),
    ("Sol, G3", 196.00),
    ("Si, B3", 246.94),
    ("Mi aigu, E4", 329.63)
])

fs_rate, signal = wavfile.read("82.407.wav")
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
freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
freqs_side = freqs[range(N//2)]  # one side frequency range

# Trouver la plus haute intensité de son et trouver sa fréquence
index = np.where(FFT_side == np.amax(FFT_side))

the_note = round(freqs_side[index[0]], 2)

high = None
low = None
high_ecart = None
low_ecart = None
found = False

for note, freq in strings.items():
    if the_note < freq and found == False:
        high = (note, freq)
        found = True
    elif the_note > freq:
        low = (note, freq)

if high != None:
    high_ecart = convert(high[1], the_note)

if low != None:
    low_ecart = convert(low[1], the_note)

if high_ecart != None and low_ecart != None:
    if high_ecart < low_ecart:
        the_goal = high
    else:
        the_goal = low

if high_ecart != None and low_ecart == None:
    the_goal = high

if low_ecart != None and high_ecart == None:
    the_goal = low


print("Note : ", the_goal[0])
if convert(the_note, the_goal[1]) > 0.1:
    print("Descendez la corde", convert(the_goal[1], the_note))
elif convert(the_note, the_goal[1]) < 0.1 and convert(the_note, the_goal[1]) > -0.1:
    print("OK", convert(the_note, the_goal[1]))
elif convert(the_note, the_goal[1]) < -0.1:
    print("Montez la corde", convert(the_goal[1],the_note))


# plotting the positive fft spectrum
# p3 = plt.plot(freqs_side, abs(FFT_side), "b")
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Count single-sided')


