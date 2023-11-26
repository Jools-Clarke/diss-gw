# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 20:42:33 2022

@author: jdcjo
"""
import numpy as np
from tqdm import tqdm #this puts loading bars onto for loops 

def blip(seed = None):
  if seed == None:
    seed = np.random.randint(0,999999)
  np.random.seed(seed)


  t = np.linspace(0,1,1024) #create an array of time between 0 and 1 second with 1024 samples
  a = np.zeros_like(t) #create an array of zeros to populate with amplitude values
  A,B,C,D,b,c = 0,0.01,((np.random.rand()-0.5)),0,0.03 + (0.05*(np.random.rand()-0.5)),0.5 + (0.5*(np.random.rand()-0.5))

  randomA = (0.5*(np.random.rand()-0.5)) #maximum amplitude

  for i,t_ in enumerate(t):
    A = randomA * np.e**( -((t_-c)**2 / b) ) #setting amplidude to bell curve
    a[i] = A * np.sin( (1/B) * (t_ - C) ) + D

  return {"data":{"time":t,"amplitude":a},"seed":seed}

def wave(seed = None):
  if seed == None:
    seed = np.random.randint(0,999999)
  np.random.seed(seed)


  t = np.linspace(0,1,1024) #create an array of time between 0 and 1 second with 1024 samples
  a = np.zeros_like(t) #create an array of zeros to populate with amplitude values

  A,B,C,D,b,c,d = 0,0.01,((np.random.rand()-0.5)),0,2 + ((np.random.rand()-0.5)),0.005 + (0.001*(np.random.rand()-0.5)),5

  randomA = (0.5*(np.random.rand()-0.5)) #maximum amplitude

  for i,t_ in enumerate(t):
    if 0<t_<0.75+C/2:
      A = randomA * 2*t_**5 # amplitude increases with polynomial 5
      B = 0.005/t_**5 #freq increases
    else:
      A = 0

    a[i] = A * np.sin( (1/B) * (t_ - C) ) + D

  return {"data":{"time":t,"amplitude":a},"seed":seed}


x_train = np.zeros([90000,1024])
x_test = np.zeros([10000,1024])

print(f"generating train data, expect {len(x_train)} vals")
for i, _ in tqdm(enumerate(x_train), total = len(x_train)):
  x_train[i] = blip()["data"]["amplitude"]

print(f"generating test data, expect {len(x_test)} vals")
for i, _ in tqdm(enumerate(x_test), total = len(x_test)):
  x_test[i] = blip()["data"]["amplitude"]
  
np.savez_compressed("test_blip_data", blip_train = x_train, blip_test = x_test )

x_train = np.zeros([90000,1024])
x_test = np.zeros([10000,1024])

print(f"generating train data, expect {len(x_train)} vals")
for i, _ in tqdm(enumerate(x_train), total = len(x_train)):
  x_train[i] = wave()["data"]["amplitude"]

print(f"generating test data, expect {len(x_test)} vals")
for i, _ in tqdm(enumerate(x_test), total = len(x_test)):
  x_test[i] = wave()["data"]["amplitude"]
  
np.savez_compressed("test_wave_data", blip_train = x_train, blip_test = x_test )

  