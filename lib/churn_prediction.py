import torch
import torch.nn as nn
import datetime
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import itertools

from typing import List, Dict, TypeVar, Union
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold

from lib.data_preprocessor import NNDataPreprocess
from lib.model import NNModel
from lib.early_stopping import EarlyStopping
from lib.chart_visualizer import ChartVisualizer
from lib.parameter_selector import ParmasSelector

model = TypeVar(torch.nn.modules.module.Module)


class ChurnPrediction:

    def __init__(self, df_all_data, is_display_detail=True, is_display_batch_info=False,):

        self.is_display_detail = is_display_detail
        self.is_display_batch_info = is_display_batch_info

        self._NNDataP = NNDataPreprocess(df_all_data, test_fraction=0.2)
        self._chart_visual = ChartVisualizer()
        self._parmas_selector = ParmasSelector()

        # init variable
        self.__df_all_combinations = None

        # __________init variable__________

        # _______all tuning parmas_______
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

        # _______other variable_______
        self.cv_num = None
        self.df_best_model_cv_perf = None

        self.nn_model = None
        self.best_model = None
        self.dict_best_parmas_to_test_model = None
        # remark: first int: model_index, second int: cv_index (cv_num - 1), model: model, float: cv_loss
        self.dict_cv_model_and_loss: Dict[int, Dict[int, Union[model, float]]] = {}

        self.parameters = None

        self.num_max_epochs = None
        self.early_stopping = None

        self.is_log_in_tsboard = None
        self.log_dir = None

        self.list_train_acc = None
        self.list_train_loss = None
        self.list_valid_acc = None
        self.list_valid_loss = None
        self.list_test_acc = None
        self.list_test_loss = None

        self.ts_test_pred_label = None

        print("ChurnPrediction object created")

    # _______show and edit parmas setting_______

    def select_parmas(self, str_selection):

        self.parameters = self._parmas_selector.select(str_selection)
        self.__prepare_tuning_combination()

    def show_available_parmas_options(self):

        print(self._parmas_selector.show_available_parmas_options())

    def __prepare_tuning_combination(self):

        self.list_all_combinations = list(self.product_dict(**self.parameters))
        self.__df_all_combinations = pd.DataFrame(self.list_all_combinations)
        self.drop_parmas_combinations()
        print(f'number of combinations: {len(self.__df_all_combinations)}')
        display(self.__df_all_combinations)

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

    # _______visualize setting and performance_______

    def preview_model(self):

        return print(self.nn_model)

    def show_label_distribution(self):

        df_label_pie_chart = self._NNDataP._prepare_plot_pie_ftr_distribution()
        self._chart_visual.plot_pie_label_distribution(df_label_pie_chart,
                                                       'counts', 'status', 'Exited and not Exited distribution')

    def show_tuning_combinations(self):

        print(f'num of combination: {len(self.__df_all_combinations)}')
        display(self.__df_all_combinations)

    # _______extract and load parmas_______

    def __extract_parmas(self, dict_parmas, is_train_model):

        if is_train_model:

            # model_parmas
            self.list_layers_input_size = dict_parmas['list_layers_input_size']
            self.dropout_percent = dict_parmas['dropout_percent']

            # oversampling weight
            self.oversampling_w = dict_parmas['oversampling_w']

            # dataloader_parmas
            self.shuffle = dict_parmas['shuffle']

        # optimizer_parmas
        self.optimizer_attr = dict_parmas['optimizer_attr']
        self.lr = dict_parmas['lr']
        self.amsgrad = dict_parmas['amsgrad']

        # loss_function_parmas
        self.class_weight = dict_parmas['class_weight']

        # dataloader_parmas
        self.batch_size = dict_parmas['batch_size']

    def __load_parmas(self, is_train_model):
        """load parmas in model, optimizer and loss function
         but not need to load parmas in dataloader and oversampling setting"""

        if is_train_model:

            self.nn_model = NNModel(embedding_size=self._NNDataP.list_categorical_embed_sizes,
                                    num_numerical_cols=len(self._NNDataP.list_col_numerical),
                                    output_size=2,
                                    list_layers_input_size=self.list_layers_input_size,
                                    dropout_percent=self.dropout_percent)

        optimizer = getattr(torch.optim, self.optimizer_attr)
        self.optimizer = optimizer(self.nn_model.parameters(), lr=self.lr, amsgrad=self.amsgrad)

        self.loss_function = nn.CrossEntropyLoss(weight=self.class_weight)

    # _______logging_______

    def __prepare_parmas_desc(self, is_train_model):
        """mark down the detail time and parmas"""

        self.str_parmas_desc = ''

        if is_train_model:
            # model_parmas
            self.str_parmas_desc += f'_do_p_{self.dropout_percent}'
            # [1,2,3,4] --> '[1,2,3,4]'
            self.str_parmas_desc += f"_ly_s_[{self.list_layers_input_size}]"

        # optimizer_parmas
        self.str_parmas_desc += f'_opt_{self.optimizer_attr}'
        self.str_parmas_desc += f'_lr_{self.lr}'
        self.str_parmas_desc += f'_amsgrad_{self.amsgrad}'

        # loss_function_parmas
        if self.class_weight is not None:
            self.str_parmas_desc += f'_cw_{self.class_weight[0]:.2f}_{self.class_weight[1]:.2f}'
        else:
            self.str_parmas_desc += '_no_cw'

        # dataloader_parmas
        self.str_parmas_desc += f'_bs_{self.batch_size}'
        self.str_parmas_desc += f'_shuffle_{self.shuffle}'

        # oversampling_weight
        self.str_parmas_desc += f'_os_w_{self.oversampling_w}'

    def __create_tsboard_writer(self):

        # Comment log_dir suffix appended to the default log_dir. If log_dir is assigned, this argument has no effect.
        if self.log_dir is not None:
            str_ymd_hms = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
            self.writer = SummaryWriter(log_dir=f'{self.log_dir}/{str_ymd_hms}{self.str_parmas_desc}')
        else:
            self.writer = SummaryWriter(comment=f'_{self.str_parmas_desc}')

    def train_model(self, str_tsboard_subgrp='Train sets'):

        self.nn_model.train()

        running_loss = 0.0
        num_correct = 0
        num_total_train = 0
        total_step = len(self._NNDataP.train_loader)

        for batch_idx, (ts_x_categ, ts_x_numer, ts_y) in enumerate(self._NNDataP.train_loader):
            self.nn_model.train()

            y_pred = self.nn_model(ts_x_categ, ts_x_numer)
            single_loss = self.loss_function(y_pred, ts_y)

            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            self.optimizer.zero_grad()
            # Propagating the error backward
            single_loss.backward()
            # Optimizing the parameters
            self.optimizer.step()

            running_loss += single_loss.item()
            _, y_pred_label = torch.max(y_pred, dim=1)
            num_correct += torch.sum(y_pred_label == ts_y).item()
            num_total_train += ts_y.size(0)
            if self.is_display_batch_info:
                if (batch_idx) % 5 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(self.epoch, self.num_max_epochs, batch_idx, total_step, single_loss.item()))

        accuracy = 100 * num_correct / num_total_train
        loss = running_loss / total_step

        if self.is_display_detail:
            print(f'Train loss: {loss:.4f}, train acc: {accuracy:.4f}')

        if self.is_log_in_tsboard:
            self.writer.add_scalar(
                f'Loss/{str_tsboard_subgrp} cv_{self.cv_num}', loss, self.epoch
            )
            self.writer.add_scalar(
                f'Accuracy/{str_tsboard_subgrp} cv_{self.cv_num}', accuracy, self.epoch
            )

        self.list_train_acc.append(accuracy)
        self.list_train_loss.append(loss)

    def valid_model(self, str_tsboard_subgrp='Validation sets'):

        running_loss = 0.0
        num_correct = 0
        num_total_valid = 0

        # reduce memory consumption
        with torch.no_grad():
            self.nn_model.eval()

            for batch_idx, (ts_x_categ, ts_x_numer, ts_y) in enumerate(self._NNDataP.valid_loader):
                y_pred = self.nn_model(ts_x_categ, ts_x_numer)
                single_loss = self.loss_function(y_pred, ts_y)

                running_loss += single_loss.item()
                _, y_pred_label = torch.max(y_pred, dim=1)
                num_correct += torch.sum(y_pred_label == ts_y).item()
                num_total_valid += ts_y.size(0)

            valid_accuracy = 100 * num_correct / num_total_valid
            valid_loss = running_loss / len(self._NNDataP.valid_loader)

            if self.is_display_detail:
                print(f'Valid loss: {valid_loss:.4f}, valid acc: {valid_accuracy:.4f}')

            if self.is_log_in_tsboard:
                self.writer.add_scalar(
                    f'Loss/{str_tsboard_subgrp} cv_{self.cv_num}', valid_loss, self.epoch
                )
                self.writer.add_scalar(
                    f'Accuracy/{str_tsboard_subgrp} cv_{self.cv_num}', valid_accuracy, self.epoch
                )

            self.list_valid_acc.append(valid_accuracy)
            self.list_valid_loss.append(valid_loss)

        # early_stopping needs the validation loss to check if it has improved,
        # and if it has, it will make a checkpoint of the current model
        self.early_stopping(self.nn_model, valid_loss)

    # find the best model by cv
    def cross_validate(self, num_max_epochs=10, patience=10, is_log_in_tsboard=True,
                       log_dir=None, cv_n_splits=5):
        """is_refit: Refit an estimator using the best found parameters on the whole dataset.
        the best parameters is considered by the best valid loss"""

        self.is_log_in_tsboard = is_log_in_tsboard
        self.log_dir = log_dir

        cv_iterator = StratifiedKFold(n_splits=cv_n_splits)

        for model_idx, row in self.__df_all_combinations.iterrows():

            self.num_max_epochs = num_max_epochs

            # reset cv number
            self.cv_num = 1

            dict_parmas = dict(row)
            self.__extract_parmas(dict_parmas, is_train_model=True)
            self.__load_parmas(is_train_model=True)

            for train_index, valid_index in cv_iterator.split(self._NNDataP.ts_categ_train_valid,
                                                              self._NNDataP.ts_output_train_valid):

                self.__prepare_parmas_desc(is_train_model=True)

                self.nn_model.parmas_desc = self.str_parmas_desc
                print('train and valid model: ', self.nn_model.parmas_desc)

                # clear previous model records
                self.list_train_acc = []
                self.list_train_loss = []
                self.list_valid_acc = []
                self.list_valid_loss = []

                if self.is_log_in_tsboard:
                    self.__create_tsboard_writer()
                self._NNDataP.prepare_cv_dataloader(train_index, valid_index,
                                                    self.batch_size, self.shuffle,
                                                    self.oversampling_w)
                # will refit model, so no need to save checkpoint
                self.early_stopping = EarlyStopping(patience=patience, verbose=self.is_display_detail,
                                                    is_save_checkpoint=True)

                for self.epoch in range(1, self.num_max_epochs + 1):

                    self.train_model()
                    self.valid_model()

                    if self.early_stopping.early_stop:
                        print(f"Early stopping at {self.epoch}")
                        break

                    if self.is_display_detail or self.is_display_batch_info:
                        print('')

                # load the last checkpoint with the best model
                self.nn_model.load_state_dict(torch.load('checkpoint.pt'))

                # backup the best model each of the cv and it loss
                if model_idx not in self.dict_cv_model_and_loss.keys():
                    self.dict_cv_model_and_loss[model_idx] = {}
                if (self.cv_num - 1) not in self.dict_cv_model_and_loss[model_idx].keys():
                    self.dict_cv_model_and_loss[model_idx][self.cv_num - 1] = {}

                # record all the model and it best loss (remark: self.cv_num - 1 = cv number index)
                self.dict_cv_model_and_loss[model_idx][self.cv_num - 1]['model'] = self.nn_model
                self.dict_cv_model_and_loss[model_idx][self.cv_num - 1]['cv_loss'] = - self.early_stopping.best_score

                self.cv_num += 1

                print('')

        print('\nAll model is trained successfully')
        self.__preprocess_cv_performance()
        self.__find_best_model_and_parma()

    def __preprocess_cv_performance(self):
        
        # extract cv loss from dictionary
        list_list_cv_loss: List[List[float]] = []
        for model_index, dict_cv_index_model_and_loss in self.dict_cv_model_and_loss.items():
            list_cv_loss = []
            for cv_index, dict_model_and_loss in dict_cv_index_model_and_loss.items():
                list_cv_loss.append(dict_model_and_loss['cv_loss'])
            list_list_cv_loss.append(list_cv_loss)

        df_cv_performance = self.__df_all_combinations.copy()

        df_cv_performance['list_cv_loss'] = list_list_cv_loss
        df_cv_performance['list_mean_cv_loss'] = df_cv_performance['list_cv_loss'].apply(lambda x: np.mean(x))
        df_cv_performance['list_std_cv_loss'] = df_cv_performance['list_cv_loss'].apply(lambda x: np.std(x))
        # 'model_index' is used to find the best parameter and the best cv number
        # the order of df_cv_performance is equal to dict_cv_model_and_loss are some
        df_cv_performance['model_index'] = [model_index for model_index in self.dict_cv_model_and_loss.keys()]
        df_cv_performance['best_cv_index'] = df_cv_performance['list_cv_loss'].apply(lambda x: x.index(min(x)))
        
        self.df_cv_performance = df_cv_performance.sort_values('list_mean_cv_loss')

    def __find_best_model_and_parma(self):

        df_best_model_cv_perf = self.df_cv_performance.head(1)

        best_model_index = df_best_model_cv_perf.model_index.values[0]
        best_cv_index = df_best_model_cv_perf.best_cv_index.values[0]
        self.best_model = self.dict_cv_model_and_loss[best_model_index][best_cv_index]['model']

        # find the best parmas (some parmas are not required to test model performance)
        list_parmas_not_for_test_model = ['shuffle', 'dropout_percent', 'list_layers_input_size', 'oversampling_w']
        list_parmas_to_test_model = list(set(df_best_model_cv_perf.columns) - set(list_parmas_not_for_test_model))

        self.dict_best_parmas_to_test_model = \
            df_best_model_cv_perf[list_parmas_to_test_model].to_dict('records')[0]

    def test_model(self, dataset):

        self.__extract_parmas(self.dict_best_parmas_to_test_model, is_train_model=False)
        self.__load_parmas(is_train_model=False)

        if dataset == 'test_set':
            self._NNDataP._prepare_test_dataloader(batch_size=self.batch_size)
            dataloader = self._NNDataP.test_loader
            self.ts_test_pred_label = torch.empty(0)
            self.ts_test_label = torch.empty(0)
        elif dataset == 'train_valid_set':
            self._NNDataP._prepare_train_valid_dataloader(batch_size=self.batch_size)
            dataloader = self._NNDataP.train_valid_loader
            self.ts_train_valid_pred_label = torch.empty(0)
            self.ts_train_valid_label = torch.empty(0)

        if not hasattr(self, 'best_model'):
            raise AttributeError("object has no attribute 'best_model'" +
                                 '\nyou need to train a model or load a model')

        self.list_test_acc = []
        self.list_test_loss = []

        running_loss = 0.0
        num_correct = 0
        num_total_test = 0

        # reduce memory consumption
        with torch.no_grad():
            self.best_model.eval()

            for batch_idx, (ts_x_categ, ts_x_numer, ts_y) in enumerate(dataloader):

                y_pred = self.best_model(ts_x_categ, ts_x_numer)
                single_loss = self.loss_function(y_pred, ts_y)

                running_loss += single_loss.item()
                _, y_pred_label = torch.max(y_pred, dim=1)

                if dataset == 'test_set':
                    self.ts_test_pred_label = torch.cat((self.ts_test_pred_label, y_pred_label))
                    self.ts_test_label = torch.cat((self.ts_test_label, ts_y))
                elif dataset == 'train_valid_set':
                    self.ts_train_valid_pred_label = torch.cat((self.ts_train_valid_pred_label, y_pred_label))
                    self.ts_train_valid_label = torch.cat((self.ts_train_valid_label, ts_y))

                num_correct += torch.sum(y_pred_label == ts_y).item()
                num_total_test += ts_y.size(0)

            test_accuracy = 100 * num_correct / num_total_test
            test_loss = running_loss / len(dataloader)

            if self.is_display_detail:
                print(f'test loss: {test_loss:.4f}, test acc: {test_accuracy:.4f}')

    def show_classification_report(self, dataset):

        print("Classification report:")
        if dataset == 'test_set':
            if self.ts_test_pred_label is None:
                raise AttributeError("the attribute 'ts_test_pred_label' is empty" +
                                     '\nyou need to test model by test_set before visualize data')
            x = self.ts_test_label
            y = self.ts_test_pred_label

        elif dataset == 'train_valid_set':
            if self.ts_test_pred_label is None:
                raise AttributeError("the attribute 'ts_train_valid_label' is empty" +
                                     '\nyou need to test model by train_valid_set before visualize data')
            x = self.ts_train_valid_label
            y = self.ts_train_valid_pred_label

        return pd.DataFrame(metrics.classification_report(
            x, y, output_dict=True, target_names=['Not exited', 'Exited'])).T


