# Assignment 3: Clustering

### Machine Learning, PSAM 5020, Spring 2020 

### 0. Topic: [Option 2](https://github.com/visualizedata/ml/tree/master/ML_assignment_3/option_2)

 use [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) to cluster a set of images **on their metadata.** The measure of success is subjective; you know you have chosen the right features and number of clusters when the images in each cluster "seem" like they belong together. 

### 1. Problem: Have to look at different clusters each


### 2. Solution
#### 2.1. choose features
```python
X = data[['has_text','kinetic', 'representation','representation_semi','pl','si','va', 'te', 'co', 'or','sh']]
```
including for altoghether 11 features
such as  representation, va: value. various degree bet. black & white, te: texture , co: color (hue) , or: orientation, ranging from the vertical to the horizontal direction 

#### 2.2. set up the first cluster range: 2-20
The inertia score shows that the ideal cluster exists somewhere between 2 to 20.

![inertia](inertia.png)


```python
# fit KMeans iteratively to begin to assess the appropriate number of clusters
for i in range(1, 61):
    km = KMeans(n_clusters=i)
    km.fit(X)
    distortions.append(km.inertia_)
    
# vizualize change in inertia
plt.plot(range(1, 61), distortions, marker='+')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
```



### 3. Selected No. of clusters: 15
#### Reason
1. the silhouette scores is the highest among those different clusters, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. so here only think 2 clusters, 3 clusters,6 clusters and 15 clusters.

``` result
For n_clusters = 2 The average silhouette_score is : 0.1776378536457528
For n_clusters = 3 The average silhouette_score is : 0.17696510491477993
For n_clusters = 6 The average silhouette_score is : 0.17058155313167256
For n_clusters = 9 The average silhouette_score is : 0.16412962954870933
For n_clusters = 12 The average silhouette_score is : 0.16328424886818554
For n_clusters = 15 The average silhouette_score is : 0.17246813464737407
For n_clusters = 18 The average silhouette_score is : 0.15721887693477937
For n_clusters = 21 The average silhouette_score is : 0.1658130121781565
```

2.  The silhouette plot shows that the n_clusters value of 6 seems like  bad pick for the given data due to the presence of clusters with below average silhouette scores and also due to wide fluctuations in the size of the silhouette plots. Silhouette analysis is more ambivalent in deciding between 2,3 and 15.
![cluster6](cluster.png)

3. Also from the thickness of the silhouette plot the cluster size can be visualized. The silhouette plot for cluster 0 when n_clusters is equal to 2, is bigger in size owing to the grouping of the 3 sub clusters into one big cluster. However when the n_clusters is equal to 15, all the plots are more or less of similar thickness and hence are of similar sizes as can be also verified from the labelled scatter plot on the right.
![cluster15](cluster15.png)



### 4. Further steps:
I didn't include string feature into the cluster without further modification. If I can include information like medium, the clusters can be more informative.



### 5. some fun facts about 15 clusters:
cluster 0 : greyish clusters
![32](img_small/32_small.jpg)
![37](img_small/37_small.jpg)
![38](img_small/38_small.jpg)

cluster 1: complicated lines
![321](img_small/321_small.jpg)
![322](img_small/322_small.jpg)
![335](img_small/335_small.jpg)

cluster 2 : Jackson Pollock style
![110](img_small/110_small.jpg)
![124](img_small/124_small.jpg)
![137](img_small/137_small.jpg)

cluster 3 : blurred lines
![69](img_small/69_small.jpg)
![72](img_small/72_small.jpg)
![75](img_small/75_small.jpg)

cluster 4 : black lines
![53](img_small/53_small.jpg)
![55](img_small/55_small.jpg)
![57](img_small/57_small.jpg)

cluster 5: vertical lines
![130](img_small/130_small.jpg)
![138](img_small/138_small.jpg)
![144](img_small/144_small.jpg)

cluster 6 : black dots
![131](img_small/131_small.jpg)
![132](img_small/132_small.jpg)
![149](img_small/149_small.jpg)

cluster 7 triangles
![89](img_small/89_small.jpg)
![90](img_small/90_small.jpg)
![91](img_small/91_small.jpg)

cluster 8 black and white?
![120](img_small/120_small.jpg)
![128](img_small/128_small.jpg)
![139](img_small/139_small.jpg)

cluster 9 wooden texture(colors)
![41](img_small/41_small.jpg)
![44](img_small/44_small.jpg)
![49](img_small/49_small.jpg)

cluster 10 slightly red?
![343](img_small/343_small.jpg)
![360](img_small/360_small.jpg)
![364](img_small/364_small.jpg)

cluster 11 black with little white
![14](img_small/14_small.jpg)
![19](img_small/19_small.jpg)
![26](img_small/26_small.jpg)

cluster 12 color blocks
![174](img_small/174_small.jpg)
![176](img_small/176_small.jpg)
![188](img_small/188_small.jpg)

cluster 13 faded flower-ish stuff
![231](img_small/231_small.jpg)
![235](img_small/235_small.jpg)
![237](img_small/237_small.jpg)

cluster 14 highlight from dark series
![227](img_small/227_small.jpg)
![228](img_small/228_small.jpg)
![230](img_small/230_small.jpg)

I like the second cluster, (cluster1) which shows the features of the pictures very clearly, all completed lines tangled together. I guess the model speficy the messy-ish look as the common feature for this group


