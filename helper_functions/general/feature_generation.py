import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class NearestNeighbourFeature:
    '''
    Inspired by 1st place Kaggle winner for Optiver Volatility Prediction.
    A more general function to be used in other projects; take the nearest class
    neighbours for a particular feature in the sample space (e.g. nearest stocks by
    examining all the values of a particular feature, e.g. volatility, across time;
    the samples). Use the nearest neighbours from this function to generate a feature.
    '''
    def __init__(self,
                 features: pd.DataFrame,
                 class_label_name: str,
                 sample_index_name: str,
                 feature_name: str,
                 n_neighbours_max: int):
        '''
        :param features: the original DataFrame of features.
        :param class_label_name: the different classes which we want to group, e.g. stock.
        :param sample_index_name:  the different sample index 'name', e.g. time.
        :param feature_name: the name of the feature to be grouped.
        :param n_neighbours_max: maximum no. of nearest neighbours to be fit.
        '''

        feature_pivot = features.set_index(class_label_name).pivot(columns=sample_index_name, values=feature_name)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())

        nn = NearestNeighbors(
            n_neighbors=n_neighbours_max,
            p=2  # Euclidean distance
        )
        nn.fit(feature_pivot)
        _, neighbours = nn.kneighbors(feature_pivot, return_distance=True)

        feature_values_neighbours = np.zeros((n_neighbours_max, *feature_pivot.shape))
        for i in range(n_neighbours_max):
            # assigns the values (all sample points) of the ith nearest neighbour for each class
            feature_values_neighbours[i, :, :] += feature_pivot.values[neighbours[:, i], :]

        self.feature_values_neighbours = feature_values_neighbours
        self.columns = list(feature_pivot.columns)
        self.index = list(feature_pivot.index)
        self.class_label_name = class_label_name
        self.sample_index_name = sample_index_name
        self.feature_name = feature_name

    def create_nearest_neighbour_feature(
            self,
            num_neighbours_include: int,
            aggregate_function=np.mean,
            exclude_self: bool = True
    ) -> pd.DataFrame:
        '''
        What we are actually doing here - we take our array feature_values_neighbours, which has for each class,
        n_neighbours_max sets of the number of samples. Here we aggregate across the nearest neighbours for each class
        at each sample point, giving us a value for the feature for each class, sample pair.
        :param num_neighbours_include: number of neighbours to include in aggregate across neighbours
        :param aggregate_function: how to aggregate, e.g. mean, max, min
        :param exclude_self: should the class include itself in the nearest neighbor aggregate (self will always be first nearest neighbour)
        :return: feature in standard form (multiindex in class and sample space)
        '''
        start = 1 if exclude_self else 0
        pivot_aggs = pd.DataFrame(
                aggregate_function(self.feature_values_neighbours[start:num_neighbours_include, :, :], axis=0),
                columns=self.columns,
                index=self.index
        )

        feature_df = pivot_aggs.unstack().reset_index()
        feature_df.columns = [self.sample_index_name, self.class_label_name, f'{self.feature_name}_NearestNeighbours_{num_neighbours_include}_{aggregate_function.__name__}']
        return feature_df.set_index([self.sample_index_name, self.class_label_name])


class ClusterFeature:
    '''
    Similar to above, but performs clustering and takes cluster aggregates instead of NearestNeighbours.
    For the test set, we'd just use the cluster groupings fit in training.
    '''
    def __init__(self,
                 features: pd.DataFrame,
                 class_label_name: str,
                 sample_index_name: str,
                 feature_name: str,
                 num_clusters: int):
        '''
        :param features: the original DataFrame of features.
        :param class_label_name: the different classes which we want to group, e.g. stock.
        :param sample_index_name:  the different sample index 'name', e.g. time.
        :param feature_name: the name of the feature to be grouped.
        :param num_clusters: no. of clusters ot be fit.
        '''

        feature_pivot = features.set_index(class_label_name).pivot(columns=sample_index_name, values=feature_name)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())

        km = KMeans(
            n_clusters=num_clusters
        )
        km.fit(feature_pivot)
        clusters = km.predict(feature_pivot)

        feature_pivot['cluster'] = clusters
        self.feature_pivot = feature_pivot
        self.columns = list(feature_pivot.drop('cluster', axis=1).columns)
        self.index = list(feature_pivot.index)
        self.class_label_name = class_label_name
        self.sample_index_name = sample_index_name
        self.feature_name = feature_name
        self.clusters = clusters

    def create_cluster_feature(
            self,
            aggregate_function=np.mean
    ) -> pd.DataFrame:
        '''
        What we are actually doing here - we take our array feature_values_clusters, which has for each class,
        n_clusters sets of the number of samples. Here we aggregate across the cluster for which the point belongs to.
        :param aggregate_function: how to aggregate, e.g. mean, max, min
        :return: feature in standard form (multiindex in class and sample space)
        '''
        pivot_aggs = pd.DataFrame(
            columns=self.columns,
            index=self.index
        )
        agg_by_cluster = self.feature_pivot.groupby('cluster').apply(aggregate_function)
        for i in self.index:  # the class names
            pivot_aggs.loc[i] = agg_by_cluster.loc[self.feature_pivot.loc[i]['cluster']]

        feature_df = pivot_aggs.unstack().reset_index()
        feature_df.columns = [self.sample_index_name, self.class_label_name, f'{self.feature_name}_Cluster_{aggregate_function.__name__}']
        return feature_df.set_index([self.class_label_name, self.sample_index_name])

    def get_clusters(self):
        return self.clusters
