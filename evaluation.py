

import torch
import soundfile as sf
import numpy as np
from pystoi.stoi import stoi

# Please see this https://github.com/schmiph2/pysepm
### PESQ : not able to install in windows
"""
from pesq import pesq
def compute_PESQ(clean_signal, noisy_signal, sr=16000):
    return pesq(sr, clean_signal, noisy_signal, "wb")
"""

def compute_STOI(clean_signal, noisy_signal, sr=16000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)
first, fs = sf.read('CleanSpeech_training/clnsp1.wav')
second, fs = sf.read('out/clnsp1 (2)_generated.wav')
print(compute_STOI(first, second))