from typing import Callable

import pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline


class TrainPipelineParallel:
    def __init__(self, pre_grouped_pipeline: Pipeline, post_grouped_pipeline_func: Callable[[], Pipeline]):
        self.pre_grouped_pipeline = pre_grouped_pipeline
        self.post_grouped_pipeline_func = post_grouped_pipeline_func
        self.models = {}

    def train_pipeline_for_column(self, group_label: str, grouped_train_data: pd.DataFrame) -> (str, Pipeline):
        X = grouped_train_data.drop('target', axis=1)
        y = grouped_train_data['target']

        pipeline = self.post_grouped_pipeline_func()
        pipeline.fit(X, y)

        return group_label, pipeline

    def predict_for_column(self, group_label: str, grouped_test_data: pd.DataFrame) -> (str, pd.DataFrame):
        trained_pipeline = self.models[group_label]
        return trained_pipeline.predict(grouped_test_data)

    def train_parallel(self, X_train: pd.DataFrame, y_train: pd.DataFrame, grouping: str, n_jobs=-1):
        self.pre_grouped_pipeline.fit(X_train)
        X_train_transformed = self.pre_grouped_pipeline.transform(X_train)

        data_train = pd.merge(X_train_transformed, y_train)
        models = Parallel(n_jobs=n_jobs)(delayed(self.train_pipeline_for_column)(group_label, grouped_data)
                                         for group_label, grouped_data in data_train.groupby(grouping))
        self.models = dict(models)

    def predict_parallel(self, X_test: pd.DataFrame, grouping: str, n_jobs=-1):
        X_test_transformed = self.pre_grouped_pipeline.transform(X_test)
        predictions = Parallel(n_jobs=n_jobs)(delayed(self.train_pipeline_for_column)(group_label, grouped_data)
                                         for group_label, grouped_data in X_test_transformed.groupby(grouping))
        predictions_dict = dict(predictions)
        return pd.concat(predictions_dict.values(), keys=predictions_dict.keys(), names=[grouping])
