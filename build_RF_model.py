from dataset_preprocess import get_dataset
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from random import shuffle
import random


class RF_build():
    def __init__(self, dataset_name,  max_depth=5, n_estimators=50, curr_random_seed=0, data_split_ratio=(0.7, 0.2, 0.1)
                 , val=True):
        self.dataset_name = dataset_name
        self.data, self.x_columns, self.y_column, self.feature_types = get_dataset(self.dataset_name)
        self.data_X = self.data[self.x_columns].values
        self.data_Y = self.data[self.y_column].values
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.curr_random_seed = curr_random_seed
        self.split_ratio = data_split_ratio
        # random.seed(self.curr_random_seed)
        # np.random.seed(self.curr_random_seed)
        self.data_upper_bounds = np.max(self.data_X, axis=0)
        self.data_lower_bounds = np.min(self.data_X, axis=0)
        self.val = val


    def data_split(self):
        if self.val :
            idx = np.arange(self.data.shape[0])
            shuffle(idx)
            train_idx = idx[:int(len(idx) * self.split_ratio[0])]
            validation_idx = idx[int(len(idx) * self.split_ratio[0])
                                 :int(len(idx) * (self.split_ratio[0] + self.split_ratio[1]))]
            test_idx = idx[int(len(idx) * (self.split_ratio[0] + self.split_ratio[1])):]
            self.train_X = self.data_X[train_idx]
            self.train_Y = self.data_Y[train_idx]
            self.val_X = self.data_X[validation_idx]
            self.val_Y = self.data_Y[validation_idx]
            self.test_X = self.data_X[test_idx]
            self.test_Y = self.data_Y[test_idx]
        else:
            idx = np.arange(self.data.shape[0])
            shuffle(idx)
            train_idx = idx[:int(len(idx) * self.split_ratio)]
            test_idx = idx[int(len(idx) * self.split_ratio):]
            self.train_X = self.data_X[train_idx]
            self.train_Y = self.data_Y[train_idx]
            self.test_X = self.data_X[test_idx]
            self.test_Y = self.data_Y[test_idx]
        return

    def build_RF_model(self):
        self.data_split()
        self.RF = RandomForestClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators, random_state=1)
        self.RF.fit(self.train_X, self.train_Y)
        self.train_acc = self.RF.score(self.train_X, self.train_Y)
        self.test_acc = self.RF.score(self.test_X, self.test_Y)
        return self.RF



