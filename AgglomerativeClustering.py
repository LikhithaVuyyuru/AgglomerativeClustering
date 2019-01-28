#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data1 = pd.read_csv('GaussianClustersData.csv', header = None)
data1.head()


# In[ ]:


plt.scatter(x = data1[0],y = data1[1], cmap='rainbow')


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


kmeans3 = KMeans(n_clusters=3)


# In[ ]:


kmeans3.fit(data1)


# In[ ]:


f, (a1, a2) = plt.subplots(1, 2, sharey=True,figsize=(10,7))
a1.set_title('K Means')
a1.scatter(data1[0],data1[1],c=kmeans3.labels_,cmap='rainbow')
a2.set_title("Original")
a2.scatter(data1[0], data1[1], cmap='rainbow')


# In[ ]:


kmeans5 = KMeans(n_clusters=5)
kmeans5.fit(data1)
f, (a1, a2) = plt.subplots(1, 2, sharey=True,figsize=(10,7))
a1.set_title('K Means')
a1.scatter(data1[0],data1[1],c=kmeans5.labels_,cmap='rainbow')
a2.set_title("Original")
a2.scatter(data1[0], data1[1], cmap='rainbow')


# In[33]:


kmeans7 = KMeans(n_clusters=7)
kmeans7.fit(data1)
f, (a1, a2) = plt.subplots(1, 2, sharey=True,figsize=(10,7))
a1.set_title('K Means')
a1.scatter(data1[0],data1[1],c=kmeans7.labels_,cmap='rainbow')
a2.set_title("Original")
a2.scatter(data1[0], data1[1], cmap='rainbow')


# In[34]:


kmeans9 = KMeans(n_clusters=9)
kmeans9.fit(data1)
f, (a1, a2) = plt.subplots(1, 2, sharey=True,figsize=(10,7))
a1.set_title('K Means')
a1.scatter(data1[0],data1[1],c=kmeans9.labels_,cmap='rainbow')
a2.set_title("Original")
a2.scatter(data1[0], data1[1], cmap='rainbow')


# In[6]:


kmeans_11 = KMeans(n_clusters=11)
kmeans_11.fit(dataset)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(dataset[0],dataset[1],c=kmeans_11.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(dataset[0], dataset[1], cmap='rainbow')


# In[36]:


kmeans_13 = KMeans(n_clusters=13)
kmeans_13.fit(dataset)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(dataset[0],dataset[1],c=kmeans_13.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(dataset[0], dataset[1], cmap='rainbow')


# In[37]:


kmeans_15 = KMeans(n_clusters=15)
kmeans_15.fit(dataset)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(dataset[0],dataset[1],c=kmeans_15.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(dataset[0], dataset[1], cmap='rainbow')


# In[38]:


kmeans_17 = KMeans(n_clusters=17)
kmeans_17.fit(dataset)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(dataset[0],dataset[1],c=kmeans_17.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(dataset[0], dataset[1], cmap='rainbow')


# In[39]:


kmeans_19 = KMeans(n_clusters=19)
kmeans_19.fit(dataset)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(dataset[0],dataset[1],c=kmeans_19.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(dataset[0], dataset[1], cmap='rainbow')

inertia_ : float

    Sum of squared distances of samples to their closest cluster center.

# In[11]:


sse_ = list()
bic_ = list()
x_axis = list()
for k in range(3, 20, 2):
    x_axis.append(k)
    kmeans = KMeans(n_clusters=k).fit(dataset)
    sse = kmeans.inertia_
    sse_.append((kmeans.inertia_).round(2))
    bic_.append((bic(kmeans, dataset, k, sse)).round(2))
print(x_axis)
print(sse_)
print(bic_)


# In[8]:


def bic(kmeans,dataset, k, sse):
    c = k
    n, d = dataset.shape
    bic = n*np.log(sse/n)+np.log(n)*c*(d+1)
    return bic


# In[12]:


fig, ax = plt.subplots()
line1, = ax.plot(x_axis, sse_, '--', linewidth=2, label='SSE')
line2, = ax.plot(x_axis, bic_, label='BIC')
ax.legend(loc='upper right')
plt.show()


# In[80]:


plt.plot(x_axis, sse_, label='SSE')
plt.title("SSE")
plt.grid()


# In[81]:


plt.plot(x_axis, bic_, label = 'BIC')
plt.title("BIC")
plt.grid()


# In[ ]:





# In[ ]:





# In[33]:


kmeans_11


# In[10]:


from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
X, Y = make_blobs(n_samples=500, n_features=2, centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1)
number_clusters = 11
fig, (a1, a2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
a1.set_xlim([-0.1, 1])
a1.set_ylim([0, len(data1) + (number_clusters + 1) * 10])
cluster_r = kmeans_11
cluster_labels = cluster_r.fit_predict(data1)
silhouette_avg = silhouette_score(data1, cluster_labels)
print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
sample_silhouette_values = silhouette_samples(data1, cluster_labels)

y_lower = 10
for i in range(number_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / n_clusters)
    a1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    a1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10 

a1.set_title("The silhouette plot for the various clusters.")
a1.set_xlabel("The silhouette coefficient values")
a1.set_ylabel("Cluster label")
a1.axvline(x=silhouette_avg, color="red", linestyle="--")

a1.set_yticks([])
a1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

colors = cm.nipy_spectral(cluster_labels.astype(float) / number_clusters)
a2.scatter(data1[0], data1[1], marker='.', s=30, lw=0, alpha=0.7,
            c=colors, edgecolor='k')
centers = cluster_r.cluster_centers_
a2.scatter(centers[:, 0], centers[:, 1], marker='o',
            c="white", alpha=1, s=200, edgecolor='k')

for i, c in enumerate(centers):
    a2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                s=50, edgecolor='k')

a2.set_title("The visualization of the clustered data.")

plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
              "with n clusters = %d" % number_clusters),
             fontsize=16, fontweight='bold')

plt.show()


# In[1]:


from scipy.cluster.hierarchy import dendrogram, linkage


# In[7]:


Z = linkage(dataset, 'single')
fig = plt.figure(figsize=(25, 25))
dn = dendrogram(Z)
plt.show()


# In[11]:


import scipy.cluster.hierarchy as sch
plt.figure(figsize=(70, 70))  
dendro = sch.dendrogram(shc.linkage(data1, method='single'))
plt.show()
plt.savefig("dendro.jpeg")


# In[12]:


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=11, affinity='euclidean', linkage='single')  
cluster.fit_predict(dataset) 
plt.figure(figsize=(10, 7))  
plt.scatter(dataset[0], dataset[1], c=cluster.labels_, cmap='rainbow')
plt.show()


# In[16]:


from sklearn.cluster import AgglomerativeClustering
import numpy as np
cluster = AgglomerativeClustering(n_clusters=11, linkage='single').fit(dataset)
plt.figure(figsize=(10, 7))  
plt.scatter(dataset[0], dataset[1], c=cluster.labels_, cmap='rainbow')
plt.show()


# In[17]:


Z = linkage(dataset, 'complete')
fig = plt.figure(figsize=(25, 25))
dn = dendrogram(Z)
plt.show()


# In[18]:


from sklearn.cluster import AgglomerativeClustering
import numpy as np
cluster = AgglomerativeClustering(n_clusters=11, linkage='complete').fit(dataset)
plt.figure(figsize=(10, 7))  
plt.scatter(dataset[0], dataset[1], c=cluster.labels_, cmap='rainbow')
plt.show()


# In[ ]:




