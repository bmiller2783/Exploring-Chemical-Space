import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class assess():
    def __init__(self):
        pass

    def davies_bouldin_score():
        from sklearn.metrics import davies_bouldin_score

    def kelley_penalty():
        pass

    def silhouette_score():
        from sklearn.metrics import silhouette_score

    def rand_score():
        from sklearn.metrics import rand_score

    def fowlkes_mallows_score():
        from sklearn.metrics import fowlkes_mallows_score

    def cosime_similarity():
        from sklearn.metrics.pairwise import cosine_similarity

    def euclidean_distance():
        from sklearn.metrics.pairwise import euclidean_distances

    def haversine_distance():
        from sklearn.metrics.pairwise import haversine_distances

    def manhattan_distance():
        from sklearn.metrics.pairwise import manhattan_distances

class Cluster():
    def __init__(self):
        super().__init__()
        self.n_clusters = 5
        self.random_state = 42
        self.labels = None
        self.eps = 3
        self.min_samples = 2

    def kmeans(self, df):
        from sklearn.cluster import KMeans
        embed = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(df)
        self.labels = embed.labels_

    def dbscan(self, df):
        from sklearn.cluster import DBSCAN
        embed = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(df)
        self.labels = embed.labels_

    def hdbscan(self, df):
        #import hdbscan as HDBSCAN
        from sklearn.cluster import HDBSCAN
        embed = HDBSCAN(cluster_selection_epsilon=self.eps, min_samples=self.min_samples).fit(df)
        self.labels = embed.labels_

    def ward(self, df):
        from sklearn.cluster import ward_tree as Ward
        embed = Ward().fit(df)
        self.labels = embed.labels_

    def birch(self, df):
        from sklearn.cluster import Birch
        embed = Birch().fit(df)
        self.labels = embed.labels_


    def get_clusters(self, df_sc, method='kmeans'):
        if method == 'kmeans':
            self.kmeans(df_sc)
        elif method == 'dbscan':
            self.dbscan(df_sc)
        elif method == 'hdbscan':
            self.hdbscan(df_sc)
        else:
            print('unknown clustering method')

class ChemicalSpace():
    def __init__(self):
        super().__init__()
        self.n_components = 2
        self.init = 'random'
        self.random_seed = 42
        self.learning_rate = 'auto'
        self.perplexity = 3
        self.k_nn = 10
        self.min_dist = 1

    def pca(self, df_sc):
        from sklearn.decomposition import PCA
        df_tf = PCA(n_components=self.n_components).fit_transform(df_sc)
        self.df = pd.DataFrame({'x':df_tf[:,0], 'y':df_tf[:,1]})

    def umap(self, df_sc):
        #from sklearn.decomposition import PCA
        from umap.umap_ import UMAP, nearest_neighbors
        #pre_df = PCA(n_components=self.n_components).fit_transform(df_sc)
        #precomputed_knn = nearest_neighbors(pre_df, n_neighbors = 500, metric="cosine", metric_kwds=None,angular=False, random_state=42)
        df_tf = UMAP(n_neighbors=self.k_nn, min_dist=self.min_dist, metric="cosine").fit_transform(df_sc)#precomputed_knn=precomputed_knn
        self.df = pd.DataFrame({'x':df_tf[:,0], 'y':df_tf[:,1]})

    def tsne(self, df_sc):
        from sklearn.manifold import TSNE
        df_tf = TSNE(n_components=self.n_components, perplexity=self.perplexity, learning_rate=self.learning_rate).fit_transform(df_sc)
        #df_tf = self.transformer.transform(df_sc)
        self.df = pd.DataFrame({'x':df_tf[:,0], 'y':df_tf[:,1]})

    def make_space(self, df_sc, method='pca'):
        if method == 'pca':
            self.pca(df_sc)
        elif method == 'umap':
            self.umap(df_sc)
        elif method == 'tsne':
            self.tsne(df_sc)
        else:
            print('unknown reduction method')
