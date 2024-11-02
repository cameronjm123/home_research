import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class NearestNeighbourFeature(BaseEstimator, TransformerMixin):
    '''
    Inspired by 1st place Kaggle winner for Optiver Volatility Prediction.
    A more general function to be used in other projects; take the nearest class
    neighbours for a particular feature in the sample space (e.g. nearest stocks by
    examining all the values of a particular feature, e.g. volatility, across time;
    the samples). Use the nearest neighbours from this function to generate a feature.
    '''
    def __init__(self,
                 class_label_name: str,
                 sample_index_name: str,
                 feature_name: str,
                 n_neighbours_max: int,
                 num_neighbours_include_in_agg: int,
                 aggregate_function=np.mean,
                 exclude_self: bool = True
                 ):
        '''
        :param class_label_name: the different classes which we want to group, e.g. stock.
        :param sample_index_name:  the different sample index 'name', e.g. time.
        :param feature_name: the name of the feature to be grouped.
        :param n_neighbours_max: maximum no. of nearest neighbours to be fit.
        :param num_neighbours_include_in_agg: number of neighbours to include in aggregate across neighbours.
        :param aggregate_function: how to aggregate, e.g. mean, max, min.
        :param exclude_self: should the class include itself in the nearest neighbor aggregate (self will always be first nearest neighbour).
        '''
        self.class_label_name = class_label_name
        self.sample_index_name = sample_index_name
        self.feature_name = feature_name
        self.n_neighbours_max = n_neighbours_max
        self.num_neighbours_include_in_agg = num_neighbours_include_in_agg
        self.aggregate_function = aggregate_function
        self.exclude_self = exclude_self
        self.neighbours = None

    def fit(self, X, y=None):
        # it is fine to be fitting the nearest neighbours as we are in the training set (no look ahead bias as this is the essence of the training set)
        feature_pivot = X.set_index(self.class_label_name).pivot(columns=self.sample_index_name, values=self.feature_name)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())  # mean of all classes at each time point

        nn = NearestNeighbors(
            n_neighbors=self.n_neighbours_max,
            p=2  # Euclidean distance
        )
        nn.fit(feature_pivot)
        _, self.neighbours = nn.kneighbors(feature_pivot, return_distance=True)
        return self

    def transform(self, X):
        feature_pivot = X.set_index(self.class_label_name).pivot(columns=self.sample_index_name, values=self.feature_name)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())  # mean of all classes at each time point

        feature_values_neighbours = np.zeros((self.n_neighbours_max, *feature_pivot.shape))
        for i in range(self.n_neighbours_max):
            # assigns the values (all sample points) of the ith nearest neighbour for each class
            feature_values_neighbours[i, :, :] += feature_pivot.values[self.neighbours[:, i], :]

        start = 1 if self.exclude_self else 0
        pivot_aggs = pd.DataFrame(
            self.aggregate_function(feature_values_neighbours[start:self.num_neighbours_include_in_agg, :, :], axis=0),
            columns=list(feature_pivot.columns),
            index=list(feature_pivot.index)
        )

        feature_df = pivot_aggs.unstack().reset_index()
        feature_df.columns = [self.sample_index_name, self.class_label_name, f'{self.feature_name}_NearestNeighbours_{self.num_neighbours_include_in_agg}_{self.aggregate_function.__name__}']
        return X.set_index([self.sample_index_name, self.class_label_name]).join(feature_df).reset_index()


class ClusterFeature:
    '''
    Similar to above, but performs clustering and takes cluster aggregates instead of NearestNeighbours.
    For the test set, we'd just use the cluster groupings fit in training.
    '''
    def __init__(self,
                 class_label_name: str,
                 sample_index_name: str,
                 feature_name: str,
                 num_clusters: int,
                 aggregate_function=np.mean):
        '''
        :param class_label_name: the different classes which we want to group, e.g. stock.
        :param sample_index_name:  the different sample index 'name', e.g. time.
        :param feature_name: the name of the feature to be grouped.
        :param num_clusters: no. of clusters ot be fit.
        :param aggregate_function: how to aggregate, e.g. mean, max, min
        '''
        self.class_label_name = class_label_name
        self.sample_index_name = sample_index_name
        self.feature_name = feature_name
        self.num_clusters = num_clusters
        self.aggregate_function = aggregate_function
        self.clusters = None

    def fit(self, X, y=None):
        # it is fine to be fitting the nearest neighbours as we are in the training set (no look ahead bias as this is the essence of the training set)
        feature_pivot = X.set_index(self.class_label_name).pivot(columns=self.sample_index_name, values=self.feature_name)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())

        km = KMeans(
            n_clusters=self.num_clusters
        )
        km.fit(feature_pivot)
        self.clusters = km.predict(feature_pivot)
        return self

    def transform(self, X):
        feature_pivot = X.set_index(self.class_label_name).pivot(columns=self.sample_index_name, values=self.feature_name)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())

        feature_pivot['cluster'] = self.clusters

        pivot_aggs = pd.DataFrame(
            columns=list(feature_pivot.drop('cluster', axis=1).columns),
            index=list(feature_pivot.index)
        )
        agg_by_cluster = feature_pivot.groupby('cluster').apply(self.aggregate_function)
        for i in feature_pivot.index:  # the class names
            pivot_aggs.loc[i] = agg_by_cluster.loc[feature_pivot.loc[i]['cluster']]

        feature_df = pivot_aggs.unstack().reset_index()
        feature_df.columns = [self.sample_index_name, self.class_label_name, f'{self.feature_name}_Cluster_{self.aggregate_function.__name__}']
        feature_df = feature_df.set_index([self.class_label_name, self.sample_index_name])
        return X.set_index([self.sample_index_name, self.class_label_name]).join(feature_df).reset_index()
