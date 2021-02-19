import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
pd.set_option('mode.chained_assignment', None)


class ChurnPredictionDataset(Dataset):
    """
    builds a parallelizable and memory saving data generator
    """

    def __init__(self, ts_x_categ, ts_x_numer, ts_y):
        self.ts_x_categ = ts_x_categ
        self.ts_x_numer = ts_x_numer
        self.ts_y = ts_y

        if (len(self.ts_x_categ) != len(self.ts_y)) or (len(self.ts_x_numer) != len(self.ts_y)):
            raise Exception("The length of X does not match the length of Y")

        print("ChurnPredictionDataset object created")

    def __len__(self):
        return len(self.ts_x_categ)

    def __getitem__(self, index):
        # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
        _ts_x_categ = self.ts_x_categ[index]
        _ts_x_numer = self.ts_x_numer[index]
        _ts_y = self.ts_y[index]

        return _ts_x_categ, _ts_x_numer, _ts_y


class NNDataPreprocess:

    def __init__(self, df_all_data, test_fraction, seed, is_stratify):
        self.df_all_data = df_all_data

        self.__declare_interested_col_type()
        self.__preprocess_categorical_col()
        self.__get_embedding_sizes()

        self.__train_test_split(test_size=test_fraction, random_state=seed, is_stratify=is_stratify)

        # __________init variable__________
        self._subset_ts_categ_train = None
        self._subset_ts_categ_valid = None

        self._subset_ts_numer_train = None
        self._subset_ts_numer_valid = None

        self._subset_ts_output_train = None
        self._subset_ts_output_valid = None

        self.weighted_sampler = None

    def __declare_interested_col_type(self):
        """only consider the interested col in the function
        and define the types of col --> affect the step of preprocessing"""

        self.list_col_categorical = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
        self.list_col_numerical = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        self.list_col_outputs = ['Exited']

    def __preprocess_categorical_col(self):
        self.df_all_data[self.list_col_categorical] = self.df_all_data[self.list_col_categorical].astype('category')
        self.list_col_converted_categorical = self.__convert_categorical_col_to_int()

    def __get_embedding_sizes(self):
        list_categorical_column_sizes = [len(self.df_all_data[col].cat.categories) for col in
                                         self.list_col_categorical]
        self.list_categorical_embed_sizes = [(col_size, min(50, (col_size + 1) // 2)) for col_size in
                                             list_categorical_column_sizes]

    def __convert_categorical_col_to_int(self, prefix='converted_'):
        """convert categorical data to integer data"""

        list_added_col = []

        for col in self.list_col_categorical:
            list_col_converted_categorical = f'{prefix}{col}'
            self.df_all_data[list_col_converted_categorical] = self.df_all_data[col].cat.codes.values

            list_added_col.append(list_col_converted_categorical)

        return list_added_col

    def __train_test_split(self, test_size, random_state, is_stratify):
        """is_stratify --> split the train test stratify"""

        # categorical data
        ts_categ_data = self.__convert_df_to_ts(self.df_all_data[self.list_col_converted_categorical], torch.int64)

        # numerical data
        ts_numer_data = self.__convert_df_to_ts(self.df_all_data[self.list_col_numerical], torch.float)

        # output
        ts_output_data = self.__convert_df_to_ts(self.df_all_data[self.list_col_outputs], None).flatten()

        self.ts_categ_train_valid, self.ts_categ_test, self.ts_numer_train_valid, self.ts_numer_test, \
        self.ts_output_train_valid, self.ts_output_test = train_test_split(
            ts_categ_data, ts_numer_data, ts_output_data, test_size=test_size,
            random_state=random_state,
            stratify=ts_output_data if is_stratify else None)

    def prepare_cv_dataloader(self, train_index, valid_index,
                              batch_size, shuffle,
                              oversampling_w):

        # train_valid set --> split to train and validation set
        self._subset_ts_categ_train = Subset(self.ts_categ_train_valid, train_index)
        self._subset_ts_categ_valid = Subset(self.ts_categ_train_valid, valid_index)

        self._subset_ts_numer_train = Subset(self.ts_numer_train_valid, train_index)
        self._subset_ts_numer_valid = Subset(self.ts_numer_train_valid, valid_index)

        self._subset_ts_output_train = Subset(self.ts_output_train_valid, train_index)
        self._subset_ts_output_valid = Subset(self.ts_output_train_valid, valid_index)

        if oversampling_w is not None:
            self._create_weighted_train_sampler(oversampling_w=oversampling_w)
        else:
            # no weighted
            self.weighted_sampler = None

        self._prepare_train_dataloader(batch_size, shuffle)
        self._prepare_valid_dataloader(batch_size, shuffle)

    def _create_weighted_train_sampler(self, oversampling_w):

        if oversampling_w == 'count_balance':
            array_class_count = np.unique(self._subset_ts_output_train, return_counts=True)[1]
            array_weight = 1. / array_class_count
        elif type(oversampling_w) == np.ndarray:
            array_weight = oversampling_w
        else:
            raise Exception(
                f"The oversamling_w should either be 'count_balance' or an array, but it is {type(oversampling_w)} now")

        array_samples_weight = array_weight[self._subset_ts_output_train]
        ts_samples_weight = torch.from_numpy(array_samples_weight)

        self.weighted_sampler = WeightedRandomSampler(ts_samples_weight, len(ts_samples_weight))

    def _prepare_train_dataloader(self, batch_size, shuffle):

        self.train_loader = DataLoader(ChurnPredictionDataset(
            self._subset_ts_categ_train,
            self._subset_ts_numer_train,
            self._subset_ts_output_train,
        ), batch_size=batch_size, shuffle=shuffle, sampler=self.weighted_sampler)

    def _prepare_valid_dataloader(self, batch_size, shuffle):

        self.valid_loader = DataLoader(ChurnPredictionDataset(
            self._subset_ts_categ_valid,
            self._subset_ts_numer_valid,
            self._subset_ts_output_valid,
        ), batch_size=batch_size, shuffle=shuffle)
        
    def _prepare_test_dataloader(self, batch_size):
        """use to compare the performance of test_model between train_valid data and test data
         so shuffle = False and sampler = None"""

        # use to test model
        self.test_loader = DataLoader(ChurnPredictionDataset(
            self.ts_categ_test,
            self.ts_numer_test,
            self.ts_output_test,
        ), batch_size=batch_size, shuffle=False, sampler=None)

    @staticmethod
    def __convert_df_to_ts(df_targeted, dtype):
        return torch.tensor(df_targeted.values, dtype=dtype)

    # __________plot chart__________

    def _prepare_plot_pie_ftr_distribution(self):

        df_output = self.df_all_data[self.list_col_outputs]
        df_output.loc[:, 'status'] = np.where(df_output.Exited == 1, 'Exit', 'Not Exit')

        df_label_pie_chart = df_output.groupby('status').agg({'status': 'count'}).rename(columns={'status': 'counts'})
        df_label_pie_chart = df_label_pie_chart.reset_index()

        return df_label_pie_chart