import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
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

    def __init__(self, df_all_data, test_fraction=0.2, valid_fraction=0.2):
        self.df_all_data = df_all_data
        self.test_fraction = test_fraction
        self.valid_fraction = valid_fraction

        self.__declare_interested_col_type()
        self.__preprocess_categorical_col()
        self.__get_embedding_sizes()

        self.__train_test_split()



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

    # def _prepare_train_valid_dataloader(self, batch_size, shuffle):
    #
    #     # training set: train model, validation set: valid model
    #     # find best parmas
    #     self.train_loader = DataLoader(ChurnPredictionDataset(
    #         self.ts_train_categ_data,
    #         self.ts_train_numer_data,
    #         self.ts_train_output_data,
    #     ), batch_size=batch_size, shuffle=shuffle)
    #
    #     # training set: train model, validation set: valid model
    #     # find best parmas
    #     self.valid_loader = DataLoader(ChurnPredictionDataset(
    #         self.ts_valid_categ_data,
    #         self.ts_valid_numer_data,
    #         self.ts_valid_output_data,
    #     ), batch_size=batch_size, shuffle=shuffle)

    def _prepare_test_dataloader(self, batch_size):

        # use to test model
        self.test_loader = DataLoader(ChurnPredictionDataset(
            self.ts_categ_test,
            self.ts_numer_test,
            self.ts_output_test,
        ), batch_size=batch_size)

    def __train_test_split(self, test_size=0.2, random_state=42, is_stratify=True):
        """is_stratify do not accept bool"""

        self.num_test_records = 1000
        self.num_valid_records = 1000

        # categorical data
        ts_categ_data = self.__convert_df_to_ts(self.df_all_data[self.list_col_converted_categorical], torch.int64)

        # numerical data
        ts_numer_data = self.__convert_df_to_ts(self.df_all_data[self.list_col_numerical], torch.float)

        # output
        ts_output_data = self.__convert_df_to_ts(self.df_all_data[self.list_col_outputs], None).flatten()

        self.ts_categ_train, self.ts_categ_test, self.ts_numer_train, self.ts_numer_test, \
        self.ts_output_train, self.ts_output_test = train_test_split(
            ts_categ_data, ts_numer_data, ts_output_data, test_size=test_size,
            random_state=random_state,
            stratify=ts_output_data if is_stratify else None)

    def prepare_cv_dataloader(self, train_index, valid_index, batch_size, shuffle):

        # train set --> split to train and validation set
        set_ts_categ_train = Subset(self.ts_categ_train, train_index)
        set_ts_categ_valid = Subset(self.ts_categ_train, valid_index)

        set_ts_numer_train = Subset(self.ts_numer_train, train_index)
        set_ts_numer_valid = Subset(self.ts_numer_train, valid_index)

        self.set_ts_output_train = Subset(self.ts_output_train, train_index)
        set_ts_output_valid = Subset(self.ts_output_train, valid_index)

        self.train_loader = DataLoader(ChurnPredictionDataset(
            set_ts_categ_train,
            set_ts_numer_train,
            self.set_ts_output_train,
        ), batch_size=batch_size, shuffle=shuffle)

        self.valid_loader = DataLoader(ChurnPredictionDataset(
            set_ts_categ_valid,
            set_ts_numer_valid,
            set_ts_output_valid,
        ), batch_size=batch_size, shuffle=shuffle)



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