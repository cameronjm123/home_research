import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors



def generate_nearest_neighbour_for_feature(
        features: pd.DataFrame,
        class_label_name: str,
        sample_index_name: str,
        feature_name: str,
        n_neighbours_max: int
) -> (np.array, np.array):
    '''
    Inspired by 1st place Kaggle winner for Optiver Volatility Prediction.
    A more general function to be used in other projects; take the nearest class
    neighbours for a particular feature in the sample space (e.g. nearest stocks by
    examining all the values of a particular feature, e.g. volatility, across time;
    the samples). Use the nearest neighbours from this function to generate a feature.
    :param features: the original DataFrame of features.
    :param class_label_name: the different classes which we want to group, e.g. stock.
    :param sample_index_name:  the different sample index 'name', e.g. time.
    :param feature_name: the name of the feature to be grouped.
    :param n_neighbours_max: maximum no. of nearest neighbours to be fit.
    :return: tuple of the pivoted table with the feature of choice and the nearest neighbours.
    '''
    feature_pivot = features.set_index(class_label_name).pivot(columns=sample_index_name, values=feature_name)
    feature_pivot = feature_pivot.fillna(feature_pivot.mean())

    nn = NearestNeighbors(
        n_neighbors=n_neighbours_max,
        p=2
    )
    nn.fit(feature_pivot)
    _, neighbours = nn.kneighbors(feature_pivot, return_distance=True)
    feature_values_neighbours = np.zeros((n_neighbours_max, *feature_pivot.shape))

    for i in range(n_neighbours_max):
        feature_values_neighbours[i, :, :] += feature_pivot.values[:, neighbours[:, i]]

    print('shape should be (neighbours, classes, samples) : ', feature_values_neighbours.shape)

    return feature_values_neighbours, neighbours


def create_nearest_neighbour_feature(
        feature_values_neighbours: np.array,
        neighbours: np.array,
        class_label_name: str,
        sample_index_name: str,
        feature_name: str,
        num_neighbours_include: int,
        aggregate_function=np.mean,
        exclude_self: bool=True
) -> pd.DataFrame:
    start = 1 if exclude_self else 0
    pivot_aggs = pd.DataFrame(
            aggregate_function(feature_values_neighbours[start:num_neighbours_include, :, :], axis=0),
            columns=columns,
            index=index
    )
    # probably is just better to make a class...

    dst = pivot_aggs.unstack().reset_index()
    dst.columns = ['stock_id', 'time_id', f'{feature_col}_nn{n}_test_{agg.__name__}']
    return dst