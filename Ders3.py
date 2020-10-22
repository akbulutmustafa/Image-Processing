#!/usr/bin/env python
# coding: utf-8

# In[33]:


import os
os.getcwd()
os.listdir()


# In[34]:


path = os.getcwd()
jpg_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
jpg_files


# In[35]:


import numpy as np
import matplotlib.pyplot as plt


# In[40]:


im_1=plt.imread('canakkale.jpg')
im_1.shape


# In[37]:


def get_value_from_triple(temp_1):
    return int(temp_1[0]/3+temp_1[1]/3+temp_1[2])

def get_0_1_from_triple(temp_1):
    temp = int(temp_1[0]/3+temp_1[1]/3+temp_1[2])
    if temp<110:
        return 0
    else:
        return 1


# In[38]:


def convert_rgb_to_gray(im_1):
    m,n,k = im_1.shape
    new_image=np.zeros((m,n),dtype='uint8')
    for i in range(m):
        for j in range(n):
            s=get_value_from_triple(im_1[i,j,:])
            new_image[i,j]=s
    return new_image

def convert_rgb_to_bw(im_1):
    m,n,k = im_1.shape
    new_image=np.zeros((m,n),dtype='uint8')
    for i in range(m):
        for j in range(n):
            s=get_0_1_from_triple(im_1[i,j,:])
            new_image[i,j]=s
    return new_image


# In[39]:


im_1_gray = convert_rgb_to_gray(im_1)
im_1_bw = convert_rgb_to_bw(im_1)

plt.subplot(1,2,1)
plt.imshow(im_1_gray, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(im_1_bw, cmap='gray')
plt.show()


# In[ ]:




