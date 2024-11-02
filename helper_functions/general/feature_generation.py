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
                 train_features: pd.DataFrame,
                 class_label_name: str,
                 sample_index_name: str,
                 feature_name: str,
                 n_neighbours_max: int):
        '''
        :param train_features: the original DataFrame of features. This should NOT include the test set as we fit the nearest neighbours here.
        :param class_label_name: the different classes which we want to group, e.g. stock.
        :param sample_index_name:  the different sample index 'name', e.g. time.
        :param feature_name: the name of the feature to be grouped.
        :param n_neighbours_max: maximum no. of nearest neighbours to be fit.
        '''
        self.class_label_name = class_label_name
        self.sample_index_name = sample_index_name
        self.feature_name = feature_name
        self.n_neighbours_max = n_neighbours_max

        # it is fine to be fitting the nearest neighbours as we are in the training set (no look ahead bias as this is the essence of the training set)
        feature_pivot = train_features.set_index(class_label_name).pivot(columns=sample_index_name, values=feature_name)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())  # mean of all classes at each time point

        nn = NearestNeighbors(
            n_neighbors=n_neighbours_max,
            p=2  # Euclidean distance
        )
        nn.fit(feature_pivot)
        _, self.neighbours = nn.kneighbors(feature_pivot, return_distance=True)

    def create_nearest_neighbour_feature(
            self,
            features: pd.DataFrame,
            num_neighbours_include: int,
            aggregate_function=np.mean,
            exclude_self: bool = True
    ) -> pd.DataFrame:
        '''
        What we are actually doing here - we take our array feature_values_neighbours, which has for each class,
        n_neighbours_max sets of the number of samples. Here we aggregate across the nearest neighbours for each class
        at each sample point, giving us a value for the feature for each class, sample pair.
        :param features: features (either train or test) to transform
        :param num_neighbours_include: number of neighbours to include in aggregate across neighbours
        :param aggregate_function: how to aggregate, e.g. mean, max, min
        :param exclude_self: should the class include itself in the nearest neighbor aggregate (self will always be first nearest neighbour)
        :return: feature in standard form (multiindex in class and sample space)
        '''
        feature_pivot = features.set_index(self.class_label_name).pivot(columns=self.sample_index_name, values=self.feature_name)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())  # mean of all classes at each time point

        feature_values_neighbours = np.zeros((self.n_neighbours_max, *feature_pivot.shape))
        for i in range(self.n_neighbours_max):
            # assigns the values (all sample points) of the ith nearest neighbour for each class
            feature_values_neighbours[i, :, :] += feature_pivot.values[self.neighbours[:, i], :]

        start = 1 if exclude_self else 0
        pivot_aggs = pd.DataFrame(
                aggregate_function(feature_values_neighbours[start:num_neighbours_include, :, :], axis=0),
                columns=list(feature_pivot.columns),
                index=list(feature_pivot.index)
        )

        feature_df = pivot_aggs.unstack().reset_index()
        feature_df.columns = [self.sample_index_name, self.class_label_name, f'{self.feature_name}_NearestNeighbours_{num_neighbours_include}_{aggregate_function.__name__}']
        return feature_df.set_index([self.sample_index_name, self.class_label_name])

    def get_neighbours(self):
        return self.neighbours


class ClusterFeature:
    '''
    Similar to above, but performs clustering and takes cluster aggregates instead of NearestNeighbours.
    For the test set, we'd just use the cluster groupings fit in training.
    '''
    def __init__(self,
                 train_features: pd.DataFrame,
                 class_label_name: str,
                 sample_index_name: str,
                 feature_name: str,
                 num_clusters: int):
        '''
        :param train_features: the original DataFrame of features.
        :param class_label_name: the different classes which we want to group, e.g. stock.
        :param sample_index_name:  the different sample index 'name', e.g. time.
        :param feature_name: the name of the feature to be grouped.
        :param num_clusters: no. of clusters ot be fit.
        '''
        self.class_label_name = class_label_name
        self.sample_index_name = sample_index_name
        self.feature_name = feature_name

        feature_pivot = train_features.set_index(class_label_name).pivot(columns=sample_index_name, values=feature_name)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())

        km = KMeans(
            n_clusters=num_clusters
        )
        km.fit(feature_pivot)
        self.clusters = km.predict(feature_pivot)

    def create_cluster_feature(
            self,
            features: pd.DataFrame,
            aggregate_function=np.mean
    ) -> pd.DataFrame:
        '''
        What we are actually doing here - we take our array feature_values_clusters, which has for each class,
        n_clusters sets of the number of samples. Here we aggregate across the cluster for which the point belongs to.
        :param features: features (either train or test) to transform
        :param aggregate_function: how to aggregate, e.g. mean, max, min
        :return: feature in standard form (multiindex in class and sample space)
        '''
        feature_pivot = features.set_index(self.class_label_name).pivot(columns=self.sample_index_name, values=self.feature_name)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())

        feature_pivot['cluster'] = self.clusters

        pivot_aggs = pd.DataFrame(
            columns=list(feature_pivot.drop('cluster', axis=1).columns),
            index=list(feature_pivot.index)
        )
        agg_by_cluster = feature_pivot.groupby('cluster').apply(aggregate_function)
        for i in feature_pivot.index:  # the class names
            pivot_aggs.loc[i] = agg_by_cluster.loc[feature_pivot.loc[i]['cluster']]

        feature_df = pivot_aggs.unstack().reset_index()
        feature_df.columns = [self.sample_index_name, self.class_label_name, f'{self.feature_name}_Cluster_{aggregate_function.__name__}']
        return feature_df.set_index([self.class_label_name, self.sample_index_name])

    def get_clusters(self):
        return self.clusters
