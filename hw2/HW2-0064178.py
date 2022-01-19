#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.metrics import confusion_matrix


# ## Part 2

# In[2]:


images_data = np.genfromtxt("hw02_images.csv", delimiter = ",")
labels_data = np.genfromtxt("hw02_labels.csv", delimiter = ",")


# ## Part 3

# In[3]:


train_images = images_data[:30000,:]
test_images = images_data[30000:,:]
train_label = labels_data[:30000]
test_label = labels_data[30000:]


# ## Part 4

# In[4]:


K = int(np.max(train_label))
N = train_images.shape[0]


# In[5]:


# calculate sample means
sample_means = [np.mean(train_images[train_label == (c + 1)], axis=0) for c in range(K)]

print("\033[4msample_means\033[0m \n")
for sample_mean in sample_means:
    print(sample_mean,"\n")


# In[7]:


class_sizes= np.zeros(5)
for i in range(N):
    class_sizes[int(train_label[i]-1)]+=1


# In[6]:


# calculate sample deviations 
sample_deviations = [np.std(train_images[train_label == (c + 1)], axis=0) for c in range(K)]

print("\033[4msample_deviations\033[0m \n")
for sample_deviation in sample_deviations:
    print(sample_deviation,"\n")


# In[7]:


np.shape(sample_deviations)


# In[17]:


# calculate prior probabilities
class_priors = [np.mean(train_label == (c + 1)) for c in range(K)]

print("\033[4mclass_priors\033[0m \n", class_priors)


# ## Part 5

# $g_c(x)=\log{(p(x|y=c))}+\log{(P(y=c))}$

# $g_c(x)=\log{(\dfrac{1}{\sqrt{2\pi\sigma^2_c}}.e^{-\dfrac{(x-\mu_c^2)^2}{2\sigma^2_c}}})+\log{(P(y=c))}$ 

# $g_c(x)\approx\sum^N_{i=1}{[-\dfrac{1}{2}\log{2\pi\sigma^2_c}{-\dfrac{(x-\mu_c^2)^2}{2\sigma^2_c}}]}+\log{(P(y=c))}$ 

# In[21]:


def score_def(x):
    scores = np.array([0, 0, 0, 0, 0])
    for i in range(K):
        scores[i]  = np.sum((-0.5 * ( np.log(2 * math.pi * (sample_deviations[i]**2) ))) + 
                            (-0.5 * ((x-sample_means[i])**2) /  sample_deviations[i]**2 )) + np.log(class_priors[i])
    return scores


# In[22]:


g_scores_train = [score_def(train_images[i]) for i in range(np.shape(train_images)[0])]
g_scores_test = [score_def(train_images[i]) for i in range(np.shape(test_images)[0])]


# In[35]:


def get_labels(pred,g_scores):
    for i in range(len(g_scores)):
        max_g=np.max(g_scores[i])
        if g_scores[i][0]==max_g:
            pred.append(1)
        elif g_scores[i][1]==max_g:
            pred.append(2)
        elif g_scores[i][2]==max_g:
            pred.append(3)
        elif g_scores[i][3]==max_g:
            pred.append(4)
        else :
            pred.append(5)
        
    pred_label=np.array(pred)
    return pred_label

train_pred = get_labels([],g_scores_train)
test_pred = get_labels([],g_scores_test)

#train_pred = np.argmax(g_scores_train, axis = 1)+1
#test_pred = np.argmax(g_scores_test, axis = 1)+1


# In[36]:


confusion_matrix = pd.crosstab(train_pred, train_label, rownames=['y_pred'], colnames=['y_truth'])
print("\033[4mconfusion_matrix\033[0m \n", confusion_matrix)


# In[37]:


confusion_matrix = pd.crosstab(test_pred, test_label, rownames=['y_pred'], colnames=['y_truth'])
print("\033[4mconfusion_matrix\033[0m \n", confusion_matrix)


# In[ ]:




