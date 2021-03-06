import abc
import os
import torch
import torch.nn as nn
import datetime
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import itertools
import sklearn

from typing import List, Dict, TypeVar, Union
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from lib.data_preprocessor import NNDataPreprocess
from lib.model import NNModel
from lib.early_stopping import EarlyStopping
from lib.chart_visualizer import ChartVisualizer, make_confusion_matrix
from lib.parameter_selector import ParmasSelector

model = TypeVar(torch.nn.modules.module.Module)


class ModelTrainerBase(abc.ABC):

    ####################
    # select, show and preprocess parmas setting
    # ==================

    @abc.abstractmethod
    def select_parmas(self, str_selection):
        return NotImplemented

    @abc.abstractmethod
    def show_available_parmas_options(self):
        return NotImplemented

    @staticmethod
    @abc.abstractmethod
    def product_dict(**kwargs):
        return NotImplemented

    @abc.abstractmethod
    def drop_parmas_combinations(self):
        return NotImplemented

    ####################
    # train, valid, cv, test
    # ==================

    @abc.abstractmethod
    def train_an_epoch(self, str_epoch_operation='Training'):
        return NotImplemented

    @abc.abstractmethod
    def validate_an_epoch(self, str_epoch_operation='Validation'):
        return NotImplemented

    @abc.abstractmethod
    def run_an_epoch(self, str_epoch_operation, data_iterator, test_model_dataset=None):
        return NotImplemented

    @abc.abstractmethod
    def cross_validate(self, cv_strategy: str, num_max_epochs=10, patience=10, is_log_in_tsboard=True,
                       log_dir=None, cv_n_splits=5, is_backup_best_model=True):
        return NotImplemented

    @abc.abstractmethod
    def test_model(self, dataset, str_epoch_operation='Testing'):
        return NotImplemented

    ####################
    # visualize setting and performance
    # ==================

    @abc.abstractmethod
    def preview_model(self):
        return NotImplemented

    @abc.abstractmethod
    def show_label_distribution(self):
        return NotImplemented

    @abc.abstractmethod
    def show_classification_report(self, dataset):
        return NotImplemented

    @abc.abstractmethod
    def plot_cf_matrix(self, dataset, normalize=None):
        return NotImplemented


class ModelTrainer(ModelTrainerBase):

    def __init__(self, df_all_data, is_display_detail=True, is_display_batch_info=False, seed=0, is_stratify=True):

        self.is_display_detail = is_display_detail
        self.is_display_batch_info = is_display_batch_info

        self.__init_seeds(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if os.getenv('envir') == 'uat':
            df_all_data = df_all_data.head(1000)
            print('Reminder: in uat environment\n')

        self._NNDataP = NNDataPreprocess(df_all_data, test_fraction=0.2, seed=seed, is_stratify=is_stratify)
        self._chart_visual = ChartVisualizer()
        self._parmas_selector = ParmasSelector()

        # __________init variable__________

        # _____all tuning parmas_____
        # optimizer_parmas
        self.amsgrad = None
        self.lr = None
        self.optimizer_attr = None
        # loss_function_parmas
        self.class_weight = None
        # dataloader_parmas
        self.batch_size = None
        self.shuffle = None
        # model_parmas
        self.list_layers_input_size = None
        self.dropout_percent = None

        # _____combinations_____
        self.parameters = None
        self.__df_all_combinations = None
        self.str_parmas_desc = None

        # _____logging_____
        self.is_log_in_tsboard = None
        self.log_dir = None

        # _____cv_____
        self.num_max_epochs = None
        self.early_stopping = None
        self.cv_strategy = None
        self.cv_iterator = None
        self.cv_num = None
        self.df_best_model_cv_perf = None
        # remark: first int: model_index, second int: cv_index (cv_num - 1), model: model, float: cv_loss
        self.dict_cv_model_and_loss: Dict[int, Dict[int, Union[model, float]]] = {}

        # _____model_____
        self.model = None
        self.best_model = None
        self.dict_best_parmas_to_test_model = None

        # _____testing model records_____
        self.array_epoch_y_train = None
        self.array_epoch_y_train_pred = None
        self.array_epoch_y_test = None
        self.array_epoch_y_test_pred = None
        # ______________

        print("ChurnPrediction object created")

    ####################
    # select, show and preprocess parmas setting
    # ==================

    def select_parmas(self, str_selection):

        self.parameters = self._parmas_selector.select(str_selection)

        self.__prepare_tuning_combination()
        self.__prepare_parmas_desc()

        print(f'number of combinations: {len(self.__df_all_combinations)}')
        display(self.__df_all_combinations)

    def __prepare_tuning_combination(self):

        self.list_all_combinations = list(self.product_dict(**self.parameters))
        self.__df_all_combinations = pd.DataFrame(self.list_all_combinations)
        self.drop_parmas_combinations()

    def show_available_parmas_options(self):

        print(self._parmas_selector.show_available_parmas_options())

    @staticmethod
    def product_dict(**kwargs):

        """Cartesian product of a dictionary of lists
        {"number": [1,2,3], "color": ["orange","blue"] }
        -->
        [{"number": 1, "color": "orange"},
        {"number": 1, "color": "blue"},
        {"number": 2, "color": "orange"},
        {"number": 2, "color": "blue"},
        {"number": 3, "color": "orange"},
        {"number": 3, "color": "blue"}]"""

        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    def drop_parmas_combinations(self):
        """drop unacceptable and useless parameter combination"""

        len_df_org = len(self.__df_all_combinations)

        # necessary
        # ValueError: sampler option is mutually exclusive with shuffle
        self.__df_all_combinations = self.__df_all_combinations[~(self.__df_all_combinations.shuffle &
                                                                ~(self.__df_all_combinations.oversampling_w.isnull())
                                                                  )]
        # TODO some optimizer can not use parameter
        pass

        # optional
        # TODO some combination is not good

        if len_df_org != len(self.__df_all_combinations):
            print(f'{len_df_org - len(self.__df_all_combinations)} combinations is dropped')
        # TODO warning: do not have and combination

    ####################
    # prepare, extract and load parmas
    # ==================

    def __prepare_parmas_desc(self):
        """create 'desc' column by considering all the parmas in that row"""

        dict_parmas_to_abbrev = self._parmas_selector.get_dict_parmas_to_abbrev()
        self.__df_all_combinations['desc'] = ''

        for col in self.__df_all_combinations:
            if col != 'desc':
                self.__df_all_combinations['desc'] += self.__df_all_combinations.apply(
                    lambda x: f"""{dict_parmas_to_abbrev.get(col, col)}_{x[col]}_""", axis=1) \


        self.__df_all_combinations['desc'] = self.__df_all_combinations['desc'].str.strip('_')

    def __extract_parmas(self, dict_parmas, is_train_model):

        if is_train_model:

            # oversampling weight
            self.oversampling_w = dict_parmas['oversampling_w']

            # dataloader_parmas
            self.shuffle = dict_parmas['shuffle']

        # model_parmas
        self.list_layers_input_size = dict_parmas['list_layers_input_size']
        self.dropout_percent = dict_parmas['dropout_percent']

        # optimizer_parmas
        self.optimizer_attr = dict_parmas['optimizer_attr']
        self.lr = dict_parmas['lr']
        self.amsgrad = dict_parmas['amsgrad']

        # loss_function_parmas
        self.class_weight = dict_parmas['class_weight']

        # dataloader_parmas
        self.batch_size = dict_parmas['batch_size']

    def __build_model(self):

        self.model = NNModel(embedding_size=self._NNDataP.list_categorical_embed_sizes,
                             num_numerical_cols=len(self._NNDataP.list_col_numerical),
                             output_size=2,
                             list_layers_input_size=self.list_layers_input_size,
                             dropout_percent=self.dropout_percent)

        self.model = self.model.to(self.device)

    def __load_lf_and_optim_parmas(self):
        """load parmas in loss function and optimizer"""

        optimizer = getattr(torch.optim, self.optimizer_attr)
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr, amsgrad=self.amsgrad)

        self.loss_function = nn.CrossEntropyLoss(weight=self.class_weight).to(self.device)

    ####################
    # train, valid, cv, test
    # ==================

    def train_an_epoch(self, str_epoch_operation='Training'):

        self.model.train()

        float_avg_epoch_loss, float_epoch_acc, float_batch_f1, float_batch_precis, float_batch_recall = \
            self.run_an_epoch(str_epoch_operation, self._NNDataP.train_loader)

        # _____logging_____
        if self.is_display_detail:
            self.__log_epoch_perf(str_epoch_operation, float_avg_epoch_loss, float_epoch_acc, float_batch_f1)

        if self.is_log_in_tsboard:
            self.__log_in_tsboard(str_epoch_operation, float_avg_epoch_loss, float_epoch_acc,
                                  float_batch_f1, float_batch_precis, float_batch_recall)

    def validate_an_epoch(self, str_epoch_operation='Validation'):

        self.model.eval()

        torch.set_grad_enabled(False)
        float_avg_epoch_loss, float_epoch_acc, float_batch_f1, float_batch_precis, float_batch_recall = \
            self.run_an_epoch(str_epoch_operation, self._NNDataP.valid_loader)
        torch.set_grad_enabled(True)

        # _____logging_____
        if self.is_display_detail:
            self.__log_epoch_perf(str_epoch_operation, float_avg_epoch_loss, float_epoch_acc, float_batch_f1)

        if self.is_log_in_tsboard:
            self.__log_in_tsboard(str_epoch_operation, float_avg_epoch_loss, float_epoch_acc,
                                  float_batch_f1, float_batch_precis, float_batch_recall)
        # __________

        # early_stopping needs the validation loss to check if it has improved,
        # and if it has, it will make a checkpoint of the current model
        self.early_stopping(self.model, float_avg_epoch_loss, float_batch_f1, float_batch_precis, float_batch_recall)

    def run_an_epoch(self, str_epoch_operation, data_loader, test_model_dataset=None):

        total_step = len(data_loader)

        float_epoch_loss = 0.0
        ts_epoch_y = torch.empty(0, device=self.device)
        ts_epoch_y_pred = torch.empty(0, device=self.device)

        for batch_idx, (ts_x_categ, ts_x_numer, ts_y) in enumerate(data_loader):

            ts_x_categ = ts_x_categ.to(self.device)
            ts_x_numer = ts_x_numer.to(self.device)
            ts_y = ts_y.to(self.device)

            ts_y_pred_values = self.model(ts_x_categ, ts_x_numer)
            _, ts_y_pred = torch.max(ts_y_pred_values, dim=1)

            ts_batch_loss = self.loss_function(ts_y_pred_values, ts_y)

            # backup, then find acc and f1 later
            ts_epoch_y = torch.cat((ts_epoch_y, ts_y), 0)
            ts_epoch_y_pred = torch.cat((ts_epoch_y_pred, ts_y_pred), 0)

            if str_epoch_operation == 'Training':
                # Initializing a gradient as 0 so there is no mixing of gradient among the batches
                self.optimizer.zero_grad()
                # Propagating the error backward
                ts_batch_loss.backward()
                # # Optimizing the parameters
                self.optimizer.step()

            if self.is_display_batch_info:
                if batch_idx % 5 == 0:
                    self.__log_batch_perf(batch_idx, total_step, ts_batch_loss)

            float_epoch_loss += ts_batch_loss.item()

        float_avg_epoch_loss = float_epoch_loss / total_step
        float_epoch_acc = self.cal_accuracy(ts_epoch_y_pred, ts_epoch_y)

        # cpu: copy tensor to host memory first
        # detach: the return variable do not have grad
        array_epoch_y = ts_epoch_y.cpu().detach().numpy()
        array_epoch_y_pred = ts_epoch_y_pred.cpu().detach().numpy()

        float_batch_f1, float_batch_precis, float_batch_recall = self.__find_more_scores(array_epoch_y,
                                                                                         array_epoch_y_pred)

        # back up the y and prediction when 'Testing'
        if str_epoch_operation == 'Testing':

            if test_model_dataset == 'train_set':
                self.array_epoch_y_train = array_epoch_y
                self.array_epoch_y_train_pred = array_epoch_y_pred
            elif test_model_dataset == 'test_set':
                self.array_epoch_y_test = array_epoch_y
                self.array_epoch_y_test_pred = array_epoch_y_pred

        return float_avg_epoch_loss, float_epoch_acc, float_batch_f1, float_batch_precis, float_batch_recall

    # find the best model by cv
    def cross_validate(self, cv_strategy: str, num_max_epochs=10, patience=10, is_log_in_tsboard=True,
                       log_dir=None, cv_n_splits=5):
        """is_refit: Refit an estimator using the best found parameters on the whole dataset.
        the best parameters is considered by the best valid loss"""

        self.cv_strategy = cv_strategy
        if os.getenv('envir') == 'uat':
            self.num_max_epochs = 3
            patience = 2
        else:
            self.num_max_epochs = num_max_epochs
        self.is_log_in_tsboard = is_log_in_tsboard
        self.log_dir = log_dir

        self.cv_iterator = StratifiedKFold(n_splits=cv_n_splits)

        for model_idx, row in self.__df_all_combinations.iterrows():

            self.num_max_epochs = num_max_epochs

            # reset cv number
            self.cv_num = 1

            dict_parmas = dict(row)
            self.str_parmas_desc = row['desc']
            self.__extract_parmas(dict_parmas, is_train_model=True)

            if self.is_log_in_tsboard:
                self.__create_tsboard_writer()

            print(f'model_parmas: {self.str_parmas_desc}')

            for train_index, valid_index in self.cv_iterator.split(self._NNDataP.ts_categ_train_valid,
                                                                   self._NNDataP.ts_output_train_valid):

                self._NNDataP.prepare_cv_dataloader(train_index, valid_index,
                                                    self.batch_size, self.shuffle,
                                                    self.oversampling_w)

                self.__build_model()
                self.__load_lf_and_optim_parmas()

                print(f'cv_num: {self.cv_num}')

                self.early_stopping = EarlyStopping(patience=patience, verbose=self.is_display_detail,
                                                    is_save_checkpoint=True)

                for self.epoch in range(1, self.num_max_epochs + 1):

                    self.train_an_epoch()
                    self.validate_an_epoch()

                    if self.early_stopping.early_stop:
                        print(f"Early stopping at {self.epoch}")
                        break

                    if self.is_display_detail or self.is_display_batch_info:
                        print('')

                # load the last checkpoint with the best model
                self.model.load_state_dict(torch.load('checkpoint.pt'))

                self.__backup_cv_model_info(model_idx)

                self.cv_num += 1

                print('')

        print('\nAll model is trained successfully')
        self.__preprocess_cv_performance()
        self.find_best_model_by_strategy()

    def test_model(self, dataset, str_epoch_operation='Testing'):

        if not hasattr(self, 'best_model'):
            raise AttributeError("object has no attribute 'best_model'" +
                                 '\nyou need to train a model or load a model')

        self.__extract_parmas(self.dict_best_parmas_to_test_model, is_train_model=False)

        self.model = self.best_model
        self.__load_lf_and_optim_parmas()

        if dataset == 'test_set':
            self._NNDataP._prepare_test_dataloader(self.batch_size)
            dataloader = self._NNDataP.test_loader
        # # use the training set test the model (check whether underfitting)
        elif dataset == 'train_set':
            # find past train_dataloader which is the best
            self.__prepare_best_cv_dataloader()
            dataloader = self._NNDataP.train_loader
        else:
            raise Exception(f'{dataset} is not acceptable, either train_set or test_set are acceptable')

        self.model.eval()

        torch.set_grad_enabled(False)
        _, _, _, _, _ = \
            self.run_an_epoch(str_epoch_operation, dataloader, test_model_dataset=dataset)
        torch.set_grad_enabled(True)

    ####################
    # visualize setting and performance
    # ==================

    def preview_model(self):

        return print(self.model)

    def show_label_distribution(self):

        df_label_pie_chart = self._NNDataP.prepare_plot_pie_ftr_distribution()
        self._chart_visual.plot_pie_label_distribution(df_label_pie_chart,
                                                       'counts', 'status', 'Exited and not Exited distribution')

    def show_tuning_combinations(self):

        print(f'num of combination: {len(self.__df_all_combinations)}')
        display(self.__df_all_combinations)

    def show_classification_report(self, dataset):

        print("Classification report:")
        x, y = self.__find_spec_set_records(dataset)

        return pd.DataFrame(metrics.classification_report(
            x, y, output_dict=True, target_names=['Not exited', 'Exited'])).T

    def __build_cf_matrix(self, dataset):

        x, y = self.__find_spec_set_records(dataset)

        self.array_cf_matrix = confusion_matrix(x, y)

    def plot_cf_matrix(self, dataset, normalize=None):

        self.__build_cf_matrix(dataset)

        # labels = ['True Neg','False Pos','False Neg','True Pos']
        categories = ['Not Exited', 'Exited']
        make_confusion_matrix(self.array_cf_matrix,
                              group_names=None,
                              categories=categories,
                              cmap='Blues',
                              figsize=(10, 5),
                              normalize=normalize)

    ####################
    # logging
    # ==================

    def __create_tsboard_writer(self):

        # Comment log_dir suffix appended to the default log_dir. If log_dir is assigned, this argument has no effect.
        if self.log_dir is not None:
            str_ymd_hms = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
            self.writer = SummaryWriter(log_dir=f'{self.log_dir}/{str_ymd_hms}{self.str_parmas_desc}')
        else:
            self.writer = SummaryWriter(comment=f'_{self.str_parmas_desc}')

    def __log_batch_perf(self, batch_idx, total_step, ts_batch_loss):

        print(
            f'Epochs: {self.epoch}/{self.num_max_epochs}'
            f', Step: {batch_idx}/{total_step}'
            f', Loss: {ts_batch_loss.item():.4f}'
        )

    def __log_epoch_perf(self, str_epoch_operation, float_avg_epoch_loss, float_epoch_acc, float_batch_f1):

        print(
            f'Epochs: {self.epoch}/{self.num_max_epochs}'
            f', {str_epoch_operation} loss: {float_avg_epoch_loss:.4f}'
            f', {str_epoch_operation} acc: {float_epoch_acc:.4f}'
            f', f1: {float_batch_f1:.4f}'
        )

    def __log_in_tsboard(self, str_tsboard_subgrp, float_avg_epoch_loss, float_epoch_acc,
                         float_batch_f1, float_batch_precis, float_batch_recall):

        self.writer.add_scalar(
            f'Loss/{str_tsboard_subgrp} cv_{self.cv_num}', float_avg_epoch_loss, self.epoch
        )
        self.writer.add_scalar(
            f'Accuracy/{str_tsboard_subgrp} cv_{self.cv_num}', float_epoch_acc, self.epoch
        )
        self.writer.add_scalar(
            f'f1/{str_tsboard_subgrp} cv_{self.cv_num}', float_batch_f1, self.epoch
        )
        self.writer.add_scalar(
            f'Precision/{str_tsboard_subgrp} cv_{self.cv_num}', float_batch_precis, self.epoch
        )
        self.writer.add_scalar(
            f'Recall/{str_tsboard_subgrp} cv_{self.cv_num}', float_batch_recall, self.epoch
        )

    ####################
    # find best model and parmas
    # ==================

    def find_best_model_by_strategy(self, cv_strategy=None):

        if cv_strategy is not None:
            self.cv_strategy = cv_strategy
        self.__rank_cv_performance()

        df_best_model_cv_perf = self.df_cv_performance.head(1)

        best_model_index = df_best_model_cv_perf.model_index.values[0]
        best_cv_index = df_best_model_cv_perf.best_cv_index.values[0]
        self.best_model = self.dict_cv_model_and_loss[best_model_index][best_cv_index]['model']

        # find the best parmas (some parmas are not required to test model performance)
        list_parmas_not_for_test_model = ['shuffle', 'oversampling_w']
        list_parmas_to_test_model = list(set(df_best_model_cv_perf.columns) - set(list_parmas_not_for_test_model))

        self.dict_best_parmas_to_test_model = \
            df_best_model_cv_perf[list_parmas_to_test_model].to_dict('records')[0]

    def __preprocess_cv_performance(self):

        list_list_cv_loss, list_list_cv_f1, list_list_cv_precis, list_list_cv_recall \
            = self.__extract_cv_performance_from_dict()
        self.df_cv_performance = self.__assign_cv_performance_in_df(list_list_cv_loss, list_list_cv_f1,
                                                                    list_list_cv_precis, list_list_cv_recall)

        # 'model_index' is used to find the best parameter and the best cv number
        # the order of df_cv_performance is equal to dict_cv_model_and_loss are some
        self.df_cv_performance['model_index'] = [model_index for model_index in self.dict_cv_model_and_loss.keys()]

    def __rank_cv_performance(self):

        if self.cv_strategy == 'min_loss':
            self.df_cv_performance['best_cv_index'] = self.df_cv_performance['list_cv_loss'].apply(lambda x: x.index(min(x)))
            self.df_cv_performance.sort_values('mean_cv_loss', inplace=True)
        elif self.cv_strategy == 'max_f1':
            self.df_cv_performance['best_cv_index'] = self.df_cv_performance['list_cv_f1'].apply(lambda x: x.index(max(x)))
            self.df_cv_performance.sort_values('mean_cv_f1', inplace=True, ascending=False)
        elif self.cv_strategy == 'max_precis':
            self.df_cv_performance['best_cv_index'] = self.df_cv_performance['list_cv_precis'].apply(lambda x: x.index(max(x)))
            self.df_cv_performance.sort_values('mean_cv_precis', inplace=True, ascending=False)
        elif self.cv_strategy == 'max_recall':
            self.df_cv_performance['best_cv_index'] = self.df_cv_performance['list_cv_recall'].apply(lambda x: x.index(max(x)))
            self.df_cv_performance.sort_values('mean_cv_recall', inplace=True, ascending=False)

    def __extract_cv_performance_from_dict(self):

        list_list_cv_loss: List[List[float]] = []
        list_list_cv_f1: List[List[float]] = []
        list_list_cv_precis: List[List[float]] = []
        list_list_cv_recall: List[List[float]] = []

        for model_index, dict_cv_index_model_and_loss in self.dict_cv_model_and_loss.items():

            list_cv_loss = []
            list_cv_f1 = []
            list_cv_precis = []
            list_cv_recall = []

            for cv_index, dict_model_and_loss in dict_cv_index_model_and_loss.items():

                list_cv_loss.append(dict_model_and_loss['cv_loss'])
                list_cv_f1.append(dict_model_and_loss['cv_f1'])
                list_cv_precis.append(dict_model_and_loss['cv_precis'])
                list_cv_recall.append(dict_model_and_loss['cv_recall'])

            list_list_cv_loss.append(list_cv_loss)
            list_list_cv_f1.append(list_cv_f1)
            list_list_cv_precis.append(list_cv_precis)
            list_list_cv_recall.append(list_cv_recall)

        return list_list_cv_loss, list_list_cv_f1, list_list_cv_precis, list_list_cv_recall

    def __assign_cv_performance_in_df(self, list_list_cv_loss,
                                      list_list_cv_f1, list_list_cv_precis, list_list_cv_recall):

        df_cv_performance = self.__df_all_combinations.copy()

        df_cv_performance = self.__assign_one_score_in_df(df_cv_performance, list_list_cv_loss, 'loss')

        df_cv_performance = self.__assign_one_score_in_df(df_cv_performance, list_list_cv_f1, 'f1')
        df_cv_performance = self.__assign_one_score_in_df(df_cv_performance, list_list_cv_precis, 'precis')
        df_cv_performance = self.__assign_one_score_in_df(df_cv_performance, list_list_cv_recall, 'recall')

        return df_cv_performance

    @staticmethod
    def __assign_one_score_in_df(df_cv_performance, list_list_cv_score, score_label):
        """score_label can be 'loss', 'f1' and etc."""

        df_cv_performance[f'list_cv_{score_label}'] = list_list_cv_score
        df_cv_performance[f'mean_cv_{score_label}'] = df_cv_performance[f'list_cv_{score_label}'].apply(
            lambda x: np.mean(x))
        df_cv_performance[f'std_cv_{score_label}'] = df_cv_performance[f'list_cv_{score_label}'].apply(
            lambda x: np.std(x))

        return df_cv_performance

    def __backup_cv_model_info(self, model_idx):

        # backup the best model each of the cv and it loss
        if model_idx not in self.dict_cv_model_and_loss.keys():
            self.dict_cv_model_and_loss[model_idx] = {}
        if (self.cv_num - 1) not in self.dict_cv_model_and_loss[model_idx].keys():
            self.dict_cv_model_and_loss[model_idx][self.cv_num - 1] = {}

        # record all the model and it best loss (remark: self.cv_num - 1 = cv number index)
        self.dict_cv_model_and_loss[model_idx][self.cv_num - 1]['model'] = self.model
        self.dict_cv_model_and_loss[model_idx][self.cv_num - 1]['cv_loss'] = - self.early_stopping.best_score

        self.dict_cv_model_and_loss[model_idx][self.cv_num - 1]['cv_f1'] = self.early_stopping.f1
        self.dict_cv_model_and_loss[model_idx][self.cv_num - 1]['cv_precis'] = self.early_stopping.precision
        self.dict_cv_model_and_loss[model_idx][self.cv_num - 1]['cv_recall'] = self.early_stopping.recall

    ####################
    # other
    # ==================

    @staticmethod
    def __init_seeds(seed):
        torch.manual_seed(seed)  # sets the seed for generating random numbers.
        torch.cuda.manual_seed(
            seed)  # Sets the seed for generating random numbers for the current GPU.
        # It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
        torch.cuda.manual_seed_all(
            seed)  # Sets the seed for generating random numbers on all GPUs.
        # It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

        if seed == 0:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __prepare_best_cv_dataloader(self):
        """use the best_cv_index to find back the train_dataloader"""

        cv_index = 0

        for train_index, valid_index in self.cv_iterator.split(self._NNDataP.ts_categ_train_valid,
                                                               self._NNDataP.ts_output_train_valid):

            if cv_index == self.dict_best_parmas_to_test_model['best_cv_index']:
                # use to find past train_loader only (do not need oversampling_w)
                self._NNDataP.prepare_cv_dataloader(train_index, valid_index,
                                                    self.batch_size, self.shuffle,
                                                    oversampling_w=None)
            cv_index += 1

    def __find_spec_set_records(self, dataset):

        if dataset == 'test_set':
            if self.array_epoch_y_test_pred is None:
                raise AttributeError("the attribute 'array_epoch_y_test_pred' is empty" +
                                     '\nyou need to test model by test_set before visualize data')
            x = self.array_epoch_y_test
            y = self.array_epoch_y_test_pred

        elif dataset == 'train_set':
            if self.array_epoch_y_train_pred is None:
                raise AttributeError("the attribute 'array_epoch_y_train_pred' is empty" +
                                     '\nyou need to test model by train_set before visualize data')
            x = self.array_epoch_y_train
            y = self.array_epoch_y_train_pred

        return x, y

    @staticmethod
    def cal_accuracy(ts_y_pred, ts_y):

        num_correct = torch.sum(ts_y_pred == ts_y).item()
        num_data = ts_y.size(0)

        return 100 * num_correct / num_data

    @staticmethod
    def __find_more_scores(array_epoch_y, array_epoch_y_pred):
        """find f1, precision and recall"""

        # 'macro' calculates the F1 separated by class but not using weights for the aggregation:
        # which results in a bigger penalisation when your model does not perform well with the minority classes
        float_batch_f1 = sklearn.metrics.f1_score(array_epoch_y,
                                                  array_epoch_y_pred,
                                                  labels=None, pos_label=1, average='macro',
                                                  zero_division='warn')

        float_batch_precis = sklearn.metrics.precision_score(array_epoch_y,
                                                             array_epoch_y_pred,
                                                             labels=None, pos_label=1, average='macro',
                                                             zero_division=0)

        float_batch_recall = sklearn.metrics.recall_score(array_epoch_y,
                                                          array_epoch_y_pred,
                                                          labels=None, pos_label=1, average='macro',
                                                          zero_division='warn')

        return float_batch_f1, float_batch_precis, float_batch_recall