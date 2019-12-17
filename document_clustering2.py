"""
python document_clustering2.py --no-idf --no-minibatch

http://brandonrose.org/clustering

https://stackoverflow.com/questions/43541187/how-can-i-plot-a-kmeans-text-clustering-result-with-matplotlib


=======================================
Clustering text documents using k-means
=======================================

This is an example showing how the scikit-learn can be used to cluster
documents by topics using a bag-of-words approach. This example uses
a scipy.sparse matrix to store the features instead of standard numpy arrays.

Two feature extraction methods can be used in this example:

  - TfidfVectorizer uses a in-memory vocabulary (a python dict) to map the most
    frequent words to features indices and hence compute a word occurrence
    frequency (sparse) matrix. The word frequencies are then reweighted using
    the Inverse Document Frequency (IDF) vector collected feature-wise over
    the corpus.

  - HashingVectorizer hashes word occurrences to a fixed dimensional space,
    possibly with collisions. The word count vectors are then normalized to
    each have l2-norm equal to one (projected to the euclidean unit-ball) which
    seems to be important for k-means to work in high dimensional space.

    HashingVectorizer does not provide IDF weighting as this is a stateless
    model (the fit method does nothing). When IDF weighting is needed it can
    be added by pipelining its output to a TfidfTransformer instance.

Two algorithms are demoed: ordinary k-means and its more scalable cousin
minibatch k-means.

Additionally, latent semantic analysis can also be used to reduce dimensionality
and discover latent patterns in the data. 

It can be noted that k-means (and minibatch k-means) are very sensitive to
feature scaling and that in this case the IDF weighting helps improve the
quality of the clustering by quite a lot as measured against the "ground truth"
provided by the class label assignments of the 20 newsgroups dataset.

This improvement is not visible in the Silhouette Coefficient which is small
for both as this measure seem to suffer from the phenomenon called
"Concentration of Measure" or "Curse of Dimensionality" for high dimensional
datasets such as text data. Other measures such as V-measure and Adjusted Rand
Index are information theoretic based evaluation scores: as they are only based
on cluster assignments rather than distances, hence not affected by the curse
of dimensionality.

Note: as k-means is optimizing a non-convex objective function, it will likely
end up in a local optimum. Several runs with independent random init might be
necessary to get a good convergence.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import sklearn.datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


###############################################################################
# Load some categories from the training set
#categories = [
#    'SR_FRAME_DRV',
#    'RTDEVM_PIC',
#    'RTADAPT_FWD_FRAME',
#    'NP_HAL',
#    'RT_NSE',
#    'RTDEVM_LSPIC'
#]
# Uncomment the following to do the analysis on all the categories
#categories = None

#print("Loading paths dataset for categories:")
#print(categories)

def loaddocuments():
    rootdir = 'D:\\Data\\lsi\\bbc'
    documents = []
    r = ''
    
    df = pd.read_excel('./path_module2.xlsx')
    #df = df[(df['MODULENAME'] == 'BR_AAA') | (df['MODULENAME'] == 'NP_HAL') | (df['MODULENAME'] == 'RT_NSE') | \
    #    (df['MODULENAME'] =='RTADAPT_FWD_FRAME') | (df['MODULENAME'] =='NP_HAL') | (df['MODULENAME'] =='RT_NSE')]
    j  = 0
    target = np.zeros((df.count()['MODULENAME'],), dtype=np.int64)
    categories = df['MODULENAME'].unique().tolist()
    documents = []
    for index, row in df.iterrows():
        #data = ' '.join(row['FILENAME'].split('.')[0].split('_'))
        arr = row['FILEPATH'].split('.')[0].replace('\\', '_').split('_')
        arr = list(filter(lambda x: len(x)>0, arr))
        data = ' '.join(arr)

        documents.append(data)
        target[j] = categories.index(row['MODULENAME'])
        j = j + 1

    return sklearn.datasets.base.Bunch(data=documents, target=target, target_names=categories), categories

#dataset = fetch_20newsgroups(subset='all', categories=categories,
#                             shuffle=True, random_state=42)
                                                         
dataset, categories = loaddocuments()
print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

print('44444', labels)
print('555555777', len(labels))
print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        print('3333333333', opts.n_features)
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       norm=None,
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2,
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(dataset.data)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()


###############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)

if not opts.minibatch:
    Sum_of_squared_distances = []
    K = range(1,100)
    for k in K:
        km1 = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
        km1 = km1.fit(X)
        print('Clusters={}'.format(k))
        Sum_of_squared_distances.append(km1.inertia_)


    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
y_kmeans = km.predict(X)

print("done in %0.3fs" % (time() - t0))
print('555555', len(labels))

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()


tsne_init = 'pca'  # could also be 'random'
tsne_perplexity = 20.0
tsne_early_exaggeration = 4.0
tsne_learning_rate = 1000
random_state = 1

centroids = km.cluster_centers_
model = TSNE(n_components=2, random_state=random_state, init=tsne_init, perplexity=tsne_perplexity,
         early_exaggeration=tsne_early_exaggeration, learning_rate=tsne_learning_rate)

transformed_centroids1 = model.fit_transform(centroids)
#print(centroids)
#plt.scatter(transformed_centroids[:, 0], transformed_centroids[:, 1], marker='x')
#plt.show()
if type(X) != np.ndarray:
    X = np.asarray(X.toarray())

X = np.append(X, km.cluster_centers_, axis=0)
labels = np.append(labels, range(true_k))
transformed_centroids = model.fit_transform(X)
#else:
#    print('666666666666666666')
#    transformed_centroids = model.fit_transform(X.toarray())

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    #if type(X) != np.ndarray:
    #    X = X.toarray()
    xy = transformed_centroids[class_member_mask]
    
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    #xy = transformed_centroids[class_member_mask]
    #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #         markeredgecolor='k', markersize=6)

for i, txt in enumerate(X[-len(km.cluster_centers_):]):
    plt.annotate(categories[i], (transformed_centroids[len(X) - len(km.cluster_centers_) + i][0], transformed_centroids[len(X) - len(km.cluster_centers_) + i][1]), weight='bold')
    
plt.title('Estimated number of clusters: %d' % true_k)
plt.show()
    
    
          
plt.scatter(transformed_centroids[:, 0][0:len(X) - len(km.cluster_centers_)],
    transformed_centroids[:, 1][0:len(X) - len(km.cluster_centers_)], marker='o')
    
plt.scatter(transformed_centroids[:, 0][len(X) - len(km.cluster_centers_):],
    transformed_centroids[:, 1][len(X) - len(km.cluster_centers_):], marker='x')
for i, txt in enumerate(X[-len(km.cluster_centers_):]):
    plt.annotate(categories[i], (transformed_centroids[len(X) - len(km.cluster_centers_) + i][0], transformed_centroids[len(X) - len(km.cluster_centers_) + i][1]))
centers = km.cluster_centers_

#plt.scatter(transformed_centroids1[:, 0], transformed_centroids1[:, 1], c='black', s=200, alpha=0.5);
plt.show()


MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
from sklearn.metrics.pairwise import cosine_similarity

print('9999', X)
#print('9999', km.cluster_centers_)
#for i in range(len(km.cluster_centers_)):
#X = np.append(X, km.cluster_centers_, axis=0)
#print('9999', X)
#dist = 1 - cosine_similarity(X)
#pos = mds.fit_transform(X)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

plt.scatter(pos[:, 0], pos[:, 1], marker='o')

#dist2 = 1 - cosine_similarity(km.cluster_centers_[0:4])
dist2 = km.cluster_centers_
pos2 = mds.fit_transform(dist2)

#plt.scatter(pos2[:, 0][0:4], pos2[:, 1][0:4], marker='x')
#print('444444444444', pos2)

plt.show()

if not opts.use_hashing:
    print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
