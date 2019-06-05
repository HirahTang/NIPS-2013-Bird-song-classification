#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:24:27 2019

@author: TH
"""

# =============================================================================
# bird song recognition
# =============================================================================

import numpy as np # linear algebra
import pandas as pd # data processing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) 
# will list the files in the input directory

import os
print (os.listdir("../input"))
#%%

from pathlib import Path
import matplotlib.pyplot as plt

#%%

data_dir = Path('../input')
wav_dir = data_dir/'NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV'
spect_dir = Path('./spectrograms')
spect_dir.mkdir(parents=True, exist_ok=True)
#%%
import librosa
import librosa.display

def create_spectrogram(fn_audio, fn_gram, zoom=1): 
    clip, sample_rate = librosa.load(fn_audio, sr=None) # Load audio files
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate) # Generate MFCC spectrum
    fig = plt.figure(figsize=tuple(reversed(S.shape)), dpi=1) # Create MFCC spectrum graph
    plt.gca().set_axis_off()
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    fig.savefig(fn_gram, dpi=zoom, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
#%%

create_spectrogram(wav_dir/'train/nips4b_birds_trainfile007.wav', '/tmp/007.png', 2)
plt.imshow(plt.imread('/tmp/007.png'))
plt.show()
#display(Audio(str(wav_dir/'train/nips4b_birds_trainfile015.wav')))    
#%%

def audios_to_spectrograms(from_path, to_path, folder="", from_suffix=".wav", to_suffix=".png", zoom=1):
    (to_path/folder).mkdir(parents=True, exist_ok=True)
    fns = list((from_path/folder).glob('*' + from_suffix)) # Generate MFCC spectrum graphs from audio files

    for src in fns:
        dest = to_path/folder/(src.stem + to_suffix) # Generate all the MFCC spectrum graphs
        create_spectrogram(src, dest, zoom)


#%%
        
for ds in ('train', 'test'):
    audios_to_spectrograms(wav_dir, spect_dir, ds, zoom=2)

#%%
    
