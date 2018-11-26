#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 10:59:43 2018

@author: shekhar
"""

import numpy as np
import random
import matplotlib.pyplot as plt

products = range(10000)
users = range(1000)
purchases = []
for p in range(100000):
    u = random.choice(np.array(users))
    p = random.choice(np.array(products))
    purchases.append([u,p])
    
    
#using elbow method to find the optimal number of clusters
purchases1=np.array(purchases)
from sklearn.cluster import KMeans

#within clusters sum of squares 
#elbow method is one of the best way to find out how many clusters has to be used
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(purchases1)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#fitting the Kmeans to the dataset
#n_clusters=4 Becasue the elbow method suggested it
kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(purchases1)

#finding cluster centers
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=30,c='yellow',label='centroids')

plt.scatter(purchases1[y_kmeans==0,0],purchases1[y_kmeans==0,1],s=10,c='red',label='Cluster1')

plt.scatter(purchases1[y_kmeans==1,0],purchases1[y_kmeans==1,1],s=10,c='blue',label='Cluster2')

plt.scatter(purchases1[y_kmeans==2,0],purchases1[y_kmeans==2,1],s=10,c='cyan',label='Cluster3')

'''plt.scatter(purchases1[y_kmeans==3,0],purchases1[y_kmeans==3,1],s=10,c='magenta',label='Cluster4')

plt.scatter(purchases1[y_kmeans==4,0],purchases1[y_kmeans==4,1],s=10,c='green',label='Cluster5')

plt.scatter(purchases1[y_kmeans==5,0],purchases1[y_kmeans==5,1],s=10,c='orange',label='Cluster6')

plt.scatter(purchases1[y_kmeans==6,0],purchases1[y_kmeans==6,1],s=10,c='brown',label='Cluster7')

plt.scatter(purchases1[y_kmeans==7,0],purchases1[y_kmeans==7,1],s=10,c='black',label='Cluster8')
'''


plt.title('Purchases')
plt.xlabel('Users')
plt.ylabel('Products')
plt.legend()
plt.show()