#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial as spa
import cvxopt as cvx
import pandas as pd
import scipy.spatial.distance as dt


# ## Part 2

# In[2]:


# read data into memory
images_data = np.genfromtxt("hw06_images.csv", delimiter = ",")
labels_data = np.genfromtxt("hw06_labels.csv", delimiter = ",")


# ## Part 3

# In[3]:


train_images = images_data[:1000,:]
test_images = images_data[1000:,:]
train_label = labels_data[:1000]
test_label = labels_data[1000:]

# get X and y values
X_train = train_images
y_train = train_label.astype(int)
X_test = test_images
y_test = test_label.astype(int)

# get number of samples and number of features
N_train = X_train.shape[0]
D_train = X_train.shape[1]
N_test = X_test.shape[0]
K = int(np.max(train_label)) # classes 


# In[4]:


# define Gaussian kernel function
def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D**2 / (2 * s**2))
    return(K)


# In[5]:


# calculate Gaussian kernel
s = 10
K_train = gaussian_kernel(X_train, X_train, s)
K_test = gaussian_kernel(X_test,X_train , s)
# set learning parameters
epsilon = 1e-3


# In[7]:


def OVA(C,s=10):
    f_predicteds=[]
    f_predictedsTest=[]
    for i in range(1,K+1):
        y_trainNew=np.copy(y_train)
        y_trainNew[y_trainNew != i]=-1.0
        y_trainNew[y_trainNew == i]=1.0
        
        yyK = np.matmul(y_trainNew[:,None], y_trainNew[None,:]) * K_train
    
        P = cvx.matrix(yyK)
        q = cvx.matrix(-np.ones((N_train, 1)))
        G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
        h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
        A = cvx.matrix(1.0 * y_trainNew[None,:])
        b = cvx.matrix(0.0)

        # use cvxopt library to solve QP problems
        result = cvx.solvers.qp(P, q, G, h, A, b)
        alpha = np.reshape(result["x"], N_train)
        alpha[alpha < C * epsilon] = 0
        alpha[alpha > C * (1 - epsilon)] = C

        # find bias parameter
        support_indices, = np.where(alpha != 0)
        active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
        w0 = np.mean(y_trainNew[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))
        
        # calculate predictions on training samples
        f_predicted = np.matmul(K_train, y_trainNew[:,None] * alpha[:,None]) + w0
        f_predicteds.append(f_predicted)

        # calculate predictions on training samples
        f_predictedTest = np.matmul(K_test , y_trainNew[:,None]* alpha[:,None]) + w0 
        f_predictedsTest.append(f_predictedTest)

    return f_predicteds,f_predictedsTest
    


# In[8]:


C=10
f_predicteds,f_predictedsTest=OVA(C)


# In[9]:


Y_predicted=(np.reshape(np.stack(np.transpose(f_predicteds), axis=-1), (1000, 5)))
y_predicted = np.argmax(Y_predicted, axis = 1) + 1
confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_train), y_train, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


# In[10]:


Y_predictedTest=(np.reshape(np.stack(np.transpose(f_predictedsTest), axis=-1), (4000, 5)))
y_predictedTest = np.argmax(Y_predictedTest, axis = 1) + 1
confusion_matrix2 = pd.crosstab(np.reshape(y_predictedTest, N_test), y_test, rownames = ['y_predicted'], colnames = ['y_test'])
print(confusion_matrix2)


# In[11]:


Cs=[10**(-1), 10**(0), 10**(1), 10**(2), 10**(3)]
accuracyTrains=[]
accuracyTests=[]
for C in Cs:
    f_predicteds,f_predictedsTest=OVA(C)
    Y_predicted=(np.reshape(np.stack(np.transpose(f_predicteds), axis=-1), (1000, 5)))
    y_predicted = np.argmax(Y_predicted, axis = 1) + 1
    Y_predictedTest=(np.reshape(np.stack(np.transpose(f_predictedsTest), axis=-1), (4000, 5)))
    y_predictedTest = np.argmax(Y_predictedTest, axis = 1) + 1
    
    accuracyTrain=0
    for i in range(len(y_predicted)):
        if y_predicted[i]==y_train[i]:
            accuracyTrain=accuracyTrain+1
    accuracyTrain = accuracyTrain/N_train
    accuracyTrains.append(accuracyTrain)
    
    accuracyTest=0
    for i in range(len(y_predictedTest)):
        if y_predictedTest[i]==y_test[i]:
            accuracyTest=accuracyTest+1 
    accuracyTest = accuracyTest/N_test
    accuracyTests.append(accuracyTest)


# In[12]:


parameters = np.array(Cs).astype(str)
plt.figure(figsize = (10, 10))
#print(rmse_values_train)
plt.plot(parameters, accuracyTrains, marker = ".", markersize = 10, linestyle = "-", color = "b",label='train')
plt.plot(parameters, accuracyTests, marker = ".", markersize = 10, linestyle = "-", color = "r",label='test')
plt.xlabel("Regularization parameter (C)")
plt.ylabel("Accuracy")
plt.legend(['training', 'test'])
plt.show()


# In[ ]:




