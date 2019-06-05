#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:40:48 2019

@author: TH
"""

# =============================================================================
# Padding and resizing
# =============================================================================
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import PIL.Image as pimg
from pathlib import Path
import torch
print (os.listdir("../input"))

#%%
! tar xf ../input/spectrograms.tar.bz2 # Zip all the spectrum graphs
#%%

data_dir = Path('../input')
label_dir = data_dir/'NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS'

spect_dir = Path('./spectrograms')
resized_dir = Path('./spectrograms_resized')
resized_dir.mkdir(parents = True, exist_ok = True)
#%%
for ds in ('train', 'test'): # return the histogram of spectrum sizes
    sizes = torch.Tensor([pimg.open(p).size for p in (spect_dir/ds).glob('*.png')])
    print(ds, "[max_width, min_width, height]:")
    print(list(map(lambda t:t.item(), (torch.max(sizes[:, 0]),
                                       torch.min(sizes[:, 0]),
                                       sizes[0, 1]))))
    plt.hist(sizes[:, 0])
    plt.title(f'Widths in {ds} dataset')
    plt.show()
#%%
    
def pad_repeat(image:pimg.Image, width:int): # Resize the spectrums
  if (image.width >= width): # Not changed image
    return image

  new_im = pimg.new('RGB', (width, image.height))
#  offset = (width - image.width) // 2 % image.width
  offset = 0

  # first part of the spectrum
  box = (image.width, 0, image.width, image.height)
  new_im.paste(image.crop(box))

  while offset < width: # Resizing it to the wanted size
    new_im.paste(image, (offset, 0))
    offset += image.width
  return new_im

#%%


#offset = (676 - 136) // 2 % 136
#
#offset = 0
#print ("start_p",offset)
#box_ = (0, 0, 136, 193)
#new_im_ = pimg.new('RGB', (676, 193))
#new_im_.paste(to_pad.crop(box_))
#display(new_im_)
#while offset < 676:
#    new_im_.paste(to_pad, (offset, 0))
#    display(new_im_)
#    offset += 136
#    print (offset)
#    print (new_im_.size)
#display(new_im_)

#%%
  
to_pad = pimg.open(str(spect_dir/'train'/'nips4b_birds_trainfile115.png'))
print(to_pad.size)
display(to_pad)
padded = pad_repeat(to_pad, 676)
display(padded)
print(padded.size)
#%%



def pad_resize_folder(from_path, to_path, folder=""): # Create resized graphs
  (to_path/folder).mkdir(parents=True, exist_ok=True)
  fns = list((from_path/folder).glob('*.png'))
  mw = max(map(lambda p: pimg.open(p).width, fns))
  for src in fns:
    dest = to_path/folder/src.name
    pad_repeat(pimg.open(src), mw).resize((mw, mw)).save(dest)
    
for ds in ('train', 'test'):
    pad_resize_folder(spect_dir, resized_dir, ds)
#%%
    
import random

for ds in ('train', 'test'):
    fig, axs = plt.subplots(3,3,figsize=(12,12))
    fig.suptitle(ds)
    fns = list((resized_dir/ds).glob('*.png'))
    for fn, ax in zip(random.choices(fns, k=9),
                      axs.flatten()):
        ax.imshow(plt.imread(str(fn)))
        ax.set_title(fn.stem)
        ax.axis('off')
        
#%%
for ds in ('train', 'test'): # return the histogram of spectrum sizes
    sizes = torch.Tensor([pimg.open(p).size for p in (resized_dir/ds).glob('*.png')])
    print(ds, "[max_width, min_width, height]:")
    print(list(map(lambda t:t.item(), (torch.max(sizes[:, 0]),
                                       torch.min(sizes[:, 0]),
                                       sizes[0, 1]))))
    plt.hist(sizes[:, 0])
    plt.title(f'Widths in {ds} dataset')
    plt.show()  
#%%
! tar cjf spectrograms_resized.tar.bz2 $resized_dir # Create a zipped file
! rm -r $spect_dir $resized_dir