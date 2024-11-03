from typing import Callable

import pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline


class TrainPipelineParallel:
    def __init__(self, pre_grouped_pipeline: Pipeline, post_grouped_pipeline_func: Callable[[], Pipeline], grouping: str, other_index_columns: list[str] = None):
        self.pre_grouped_pipeline = pre_grouped_pipeline
        self.post_grouped_pipeline_func = post_grouped_pipeline_func
        self.grouping = grouping
        self.index_columns = [grouping] + other_index_columns
        self.models = {}

    def train_pipeline_for_column(self, group_label: str, grouped_train_data: pd.DataFrame) -> (str, Pipeline):
        X = grouped_train_data.drop('target', axis=1)
        y = grouped_train_data['target']

        pipeline = self.post_grouped_pipeline_func()
        pipeline.fit(X, y)

        return group_label, pipeline

    def predict_for_column(self, group_label: int | str, grouped_test_data: pd.DataFrame) -> (str, pd.DataFrame):
        trained_pipeline = self.models[group_label]
        return pd.DataFrame(trained_pipeline.predict(grouped_test_data), index=grouped_test_data.index)

    def train_parallel(self, X_train: pd.DataFrame, y_train: pd.DataFrame, n_jobs=-1):
        self.pre_grouped_pipeline.fit(X_train)
        X_train_transformed = self.pre_grouped_pipeline.transform(X_train)

        print('Completed pre-train transformations')
        data_train = pd.merge(X_train_transformed, y_train)
        data_train.set_index(self.index_columns, inplace=True)
        models = Parallel(n_jobs=n_jobs)(delayed(self.train_pipeline_for_column)(group_label, grouped_data)
                                         for group_label, grouped_data in data_train.groupby(self.grouping))
        self.models = dict(models)

    def predict_parallel(self, X_test: pd.DataFrame, n_jobs=-1):
        X_test_transformed = self.pre_grouped_pipeline.transform(X_test)
        X_test_transformed.set_index(self.index_columns, inplace=True)
        predictions = Parallel(n_jobs=n_jobs)(delayed(self.predict_for_column)(group_label, grouped_data)
                                         for group_label, grouped_data in X_test_transformed.groupby(self.grouping))
        return pd.concat(predictions).rename(columns={0: 'target'})
