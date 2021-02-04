import torch
import torch.nn as nn
import datetime
import sklearn.metrics as metrics
import pandas as pd
import itertools
from torch.utils.tensorboard import SummaryWriter

from lib.data_preprocessor import NNDataPreprocess
from lib.model import NNModel
from lib.early_stopping import EarlyStopping
from lib.chart_visualizer import ChartVisualizer


class ChurnPrediction:

    def __init__(self, df_all_data, is_display_detail=True, is_display_batch_info=False,):

        self.is_display_detail = is_display_detail
        self.is_display_batch_info = is_display_batch_info

        self._NNDataP = NNDataPreprocess(df_all_data, test_fraction=0.2)
        self._chart_visual = ChartVisualizer()

        self.__declare_tuning_parmas()

        # __________init variable__________

        # _______all tuning parmas_______
        # optimizer_parmas
        self.lr = None
        # loss_function_parmas
        self.class_weight = None
        # dataloader_parmas
        self.batch_size = None
        self.shuffle = None
        # model_parmas
        self.list_layers_input_size = None
        self.dropout_percent = None

        # _______other variable_______
        self.best_model = None
        self.best_parmas_to_test_model = None
        self.nn_model = None
        self.list_backed_up_model = []

        self.num_max_epochs = None
        self.patience = None
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

    def preview_model(self):

        return print(self.nn_model)

    def __declare_tuning_parmas(self):

        parameters = dict(

            # optimizer_parmas
            lr=[.02, 0.1],

            # dataloader_parmas
            batch_size=[1000],
            shuffle=[True, False],

            # loss_function_parmas
            class_weight=[None, torch.tensor([0.8, 1])],

            # model_parmas
            dropout_percent=[0.3, 0.5],
            list_layers_input_size=[[400, 200, 100, 50], [100, 50]]
        )

        self.list_all_combinations = list(self.product_dict(**parameters))
        self.__df_all_combinations = pd.DataFrame(self.list_all_combinations)

    def show_label_distribution(self):

        df_label_pie_chart = self._NNDataP._prepare_plot_pie_ftr_distribution()
        self._chart_visual.plot_pie_label_distribution(df_label_pie_chart,
                                                       'counts', 'status', 'Exited and not Exited distribution')

    def show_tuning_combinations(self):

        print(f'num of combination: {len(self.__df_all_combinations)}')
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

    def __extract_parmas(self, dict_parmas, is_train_model):

        if is_train_model:

            # model_parmas
            self.list_layers_input_size = dict_parmas['list_layers_input_size']
            self.dropout_percent = dict_parmas['dropout_percent']

        elif not is_train_model:
            # test data not need to shuffle
            dict_parmas['shuffle'] = False

        # optimizer_parmas
        self.lr = dict_parmas['lr']
        # loss_function_parmas
        self.class_weight = dict_parmas['class_weight']
        # dataloader_parmas
        self.batch_size = dict_parmas['batch_size']
        self.shuffle = dict_parmas['shuffle']

    def __load_parmas(self, is_train_model):

        if is_train_model:

            self.nn_model = NNModel(embedding_size=self._NNDataP.list_categorical_embed_sizes,
                                    num_numerical_cols=len(self._NNDataP.list_col_numerical),
                                    output_size=2,
                                    list_layers_input_size=self.list_layers_input_size,
                                    dropout_percent=self.dropout_percent)

        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=self.lr)

        self.loss_function = nn.CrossEntropyLoss(weight=self.class_weight)

        self._NNDataP._prepare_train_valid_dataloader(batch_size=self.batch_size, shuffle=self.shuffle)


    def __prepare_parmas_desc(self, is_train_model):
        """mark down the detail time and parmas"""

        self.str_parmas_desc = ''

        if is_train_model:
            # model_parmas
            self.str_parmas_desc += f'_dropout_p_{self.dropout_percent}'
            # [1,2,3,4] --> '[1,2,3,4]'
            self.str_parmas_desc += f"_layers_size_[{','.join(str(size) for size in self.list_layers_input_size)}]"

        # optimizer_parmas
        self.str_parmas_desc += f'_lr_{self.lr}'

        # loss_function_parmas
        if self.class_weight is not None:
            self.str_parmas_desc += f'_cw_{self.class_weight[0]:.2f}_{self.class_weight[1]:.2f}'
        else:
            self.str_parmas_desc += '_no_cw'

        # dataloader_parmas
        self.str_parmas_desc += f'_bs_{self.batch_size}'
        self.str_parmas_desc += f'_shuffle_{self.shuffle}'

    def __create_tsboard_writer(self, tsboard_remark=''):

        # Comment log_dir suffix appended to the default log_dir. If log_dir is assigned, this argument has no effect.
        if self.log_dir is not None:
            str_ymd_hms = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
            self.writer = SummaryWriter(log_dir=f'{self.log_dir}/{str_ymd_hms}{tsboard_remark}{self.str_parmas_desc}')
        else:
            self.writer = SummaryWriter(comment=f'_{tsboard_remark}{self.str_parmas_desc}')

    def train_model(self, dataloader, str_tsboard_subgrp='Train sets'):

        self.nn_model.train()

        running_loss = 0.0
        num_correct = 0
        num_total_train = 0
        total_step = len(dataloader)

        for batch_idx, (ts_x_categ, ts_x_numer, ts_y) in enumerate(dataloader):
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
                f'Loss/{str_tsboard_subgrp}', loss, self.epoch
            )
            self.writer.add_scalar(
                f'Accuracy/{str_tsboard_subgrp}', accuracy, self.epoch
            )

        self.list_train_acc.append(accuracy)
        self.list_train_loss.append(loss)

    def valid_model(self, dataloader, str_tsboard_subgrp='Validation sets'):

        running_loss = 0.0
        num_correct = 0
        num_total_valid = 0

        # reduce memory consumption
        with torch.no_grad():
            self.nn_model.eval()

            for batch_idx, (ts_x_categ, ts_x_numer, ts_y) in enumerate(dataloader):
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
                    f'Loss/{str_tsboard_subgrp}', valid_loss, self.epoch
                )
                self.writer.add_scalar(
                    f'Accuracy/{str_tsboard_subgrp}', valid_accuracy, self.epoch
                )

            self.list_valid_acc.append(valid_accuracy)
            self.list_valid_loss.append(valid_loss)

        # early_stopping needs the validation loss to check if it has improved,
        # and if it has, it will make a checkpoint of the current model
        self.early_stopping(self.nn_model, valid_loss)

    # find best model
    def find_best_parmas_by_valid_loss(self, num_max_epochs=10, patience=10,
                                       is_test_model=True, is_log_in_tsboard=True,
                                       log_dir=None):
        """is_refit: Refit an estimator using the best found parameters on the whole dataset.
        the best parameters is considered by the best valid loss"""

        self.is_log_in_tsboard = is_log_in_tsboard
        self.log_dir = log_dir

        for model_idx, dict_parmas in enumerate(self.list_all_combinations):

            self.num_max_epochs = num_max_epochs
            self.patience = patience

            self.__extract_parmas(dict_parmas, is_train_model=True)
            self.__load_parmas(is_train_model=True)

            self.__prepare_parmas_desc(is_train_model=True)
            self.__create_tsboard_writer()

            self.early_stopping = EarlyStopping(patience=self.patience, verbose=self.is_display_detail)

            self.nn_model.parmas_desc = self.str_parmas_desc
            print('train and valid model: ', self.nn_model.parmas_desc)

            # clear previous model records
            self.list_train_acc = []
            self.list_train_loss = []
            self.list_valid_acc = []
            self.list_valid_loss = []

            for self.epoch in range(1, self.num_max_epochs + 1):

                self.train_model(self._NNDataP.train_loader)
                self.valid_model(self._NNDataP.valid_loader)

                if self.early_stopping.early_stop:
                    print(f"Early stopping at {self.epoch}")
                    break

                if self.is_display_detail or self.is_display_batch_info:
                    print('')

            # load the last checkpoint with the best model
            self.nn_model.load_state_dict(torch.load('checkpoint.pt'))
            # back up all the model
            self.list_backed_up_model.append(self.nn_model)

        print('\nAll model is trained successfully')
        self.__preprocess_validation_performance()
        self.__find_best_model_and_parmas()

        if is_test_model:
            self.test_model()

    def __preprocess_validation_performance(self):

        list_parmas_desc = []
        list_best_valid_loss = []

        for model in self.list_backed_up_model:
            list_parmas_desc.append(model.parmas_desc)
            list_best_valid_loss.append(model.best_valid_loss)

        df_best_valid_loss = pd.DataFrame({'parmas_desc': list_parmas_desc,
                                           'best_valid_loss': list_best_valid_loss})

        self.df_validation_performance = pd.merge(self.__df_all_combinations, df_best_valid_loss,
                                                  left_index=True, right_index=True).sort_values('best_valid_loss')

    def __find_best_model_and_parmas(self):

        # find the best model
        for model in self.list_backed_up_model:
            if model.parmas_desc == self.df_validation_performance.loc[0, 'parmas_desc']:
                self.best_model = model
        # find the best parmas
        self.best_parmas_to_test_model = self.df_validation_performance.head(1)[['lr', 'batch_size', 'class_weight']].to_dict('records')[0]

    def test_model(self):

        self._NNDataP._prepare_test_dataloader(batch_size=self.batch_size)

        if not hasattr(self, 'best_model'):
            raise AttributeError("object has no attribute 'best_model'" +
                                 '\nyou need to train a model or load a model')

        self.__extract_parmas(self.best_parmas_to_test_model, is_train_model=False)
        self.__load_parmas(is_train_model=False)

        self.__prepare_parmas_desc(is_train_model=False)

        self.list_test_acc = []
        self.list_test_loss = []
        self.ts_test_pred_label = torch.empty(0)

        running_loss = 0.0
        num_correct = 0
        num_total_test = 0

        # reduce memory consumption
        with torch.no_grad():
            self.best_model.eval()

            for batch_idx, (ts_x_categ, ts_x_numer, ts_y) in enumerate(self._NNDataP.test_loader):

                y_pred = self.best_model(ts_x_categ, ts_x_numer)
                single_loss = self.loss_function(y_pred, ts_y)

                running_loss += single_loss.item()
                _, y_pred_label = torch.max(y_pred, dim=1)
                self.ts_test_pred_label = torch.cat((self.ts_test_pred_label, y_pred_label))
                num_correct += torch.sum(y_pred_label == ts_y).item()
                num_total_test += ts_y.size(0)

            test_accuracy = 100 * num_correct / num_total_test
            test_loss = running_loss / len(self._NNDataP.test_loader)

            if self.is_display_detail:
                print(f'test loss: {test_loss:.4f}, test acc: {test_accuracy:.4f}')

            self.list_test_acc.append(test_accuracy)
            self.list_test_loss.append(test_loss)

    def show_test_set_classification_report(self):

        if self.ts_test_pred_label is None:
            raise AttributeError("the attribute 'ts_test_pred_label' is empty" +
                                 '\nyou need to test model before visualize data' +
                                 " (try to run 'churn_prediction.test_model()')")

        print("Classification report:")
        return pd.DataFrame(metrics.classification_report(
            self._NNDataP.ts_test_output_data, self.ts_test_pred_label, output_dict=True)).T