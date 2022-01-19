# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 19:50:41 2021

@author: KEREM
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa

X = np.genfromtxt("hw07_data_set.csv", delimiter = ",")
initial_centroids = np.genfromtxt("hw07_initial_centroids.csv", delimiter = ",")

N = np.shape(X)[0]
K = np.shape(initial_centroids)[0]
D = np.shape(X)[1]

def update_centroids(memberships, X):
    if memberships is None:
        # initialize centroids
        centroids = initial_centroids
    else:
        # update centroids
        centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)

def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize = 12, 
                 markerfacecolor = cluster_colors[c], markeredgecolor = "black")
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    
   
centroids = None
memberships = None

#initialization step and estimation of the intial covariance matrix and prior probabilities.
centroids = update_centroids(memberships, X)


plt.figure(figsize = (12, 6))  
plt.subplot(1, 2, 1)
plot_current_state(centroids, memberships, X)
plt.title("initial Step:")  

memberships = update_memberships(centroids, X)
plt.subplot(1, 2, 2)
plot_current_state(centroids, memberships, X)
plt.title("initial Step:")  
plt.show()

# centroids = update_centroids(memberships, X)
# memberships = update_memberships(centroids, X)

initial_classSizes = [np.sum(memberships == c) for c in range(K)]
initial_prior = [np.mean(memberships == c) for c in range(K)]
initial_cov = [(np.matmul(np.transpose(X[memberships == (c)] - initial_prior[c]), (X[memberships == (c)] - initial_prior[c])) / initial_classSizes[c]) for c in range(K)] 

print("Initial priors:\n", initial_prior, "\n\n")
print("Initial covariance matrices:\n", initial_cov, "\n\n")


# PART 3:
    
for iteration in range(1,101):
    # print("Iteration#{}:".format(iteration))

    old_centroids = centroids
    centroids = update_centroids(memberships, X)
    # if np.alltrue(centroids == old_centroids):
    #     break
    # else:
    # plt.figure(figsize = (12, 6))    
    # plt.subplot(1, 2, 1)
    # plot_current_state(centroids, memberships, X)

    old_memberships = memberships
    memberships = update_memberships(centroids, X)
    # if np.alltrue(memberships == old_memberships):
    #     plt.show()
    #     break
    # else:
    # plt.subplot(1, 2, 2)
    # plot_current_state(centroids, memberships, X)
    # plt.show()


print("Mean Vector:\n", centroids, "\n\n")


plt.figure(figsize = (12, 6))  
plot_current_state(centroids, memberships, X)
plt.title("Final Result") 
    


