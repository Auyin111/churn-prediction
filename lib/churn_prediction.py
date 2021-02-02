import torch
import torch.nn as nn
import datetime
import sklearn.metrics as metrics
import pandas as pd
import itertools
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.data_preprocessor import NNDataPreprocess
from lib.model import NNModel
from lib.early_stopping import EarlyStopping


class ChurnPrediction:

    def __init__(self, df_all_data, is_display_detail=True, is_display_batch_info = False,
                 log_dir='C:/Users/Auyin/PycharmProjects/churn-prediction/train_valid_log',
                 ):

        self.is_display_detail = is_display_detail
        self.is_display_batch_info = is_display_batch_info
        self.log_dir = log_dir

        self._NNDataP = NNDataPreprocess(df_all_data, test_fraction=0.2)

        self.__declare_tunning_parmas()



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


        self.nn_model = None
        self.list_train_valid_model = []

        self.num_epochs = None
        self.patience = None

        self.total_step = None

        self.list_train_acc = None
        self.list_train_loss = None
        self.list_valid_acc = None
        self.list_valid_loss = None
        self.list_test_acc = None
        self.list_test_loss = None
        self.ts_test_pred_label = None


    def preview_model(self):

        return print(self.nn_model)

    def __declare_tunning_parmas(self):

        # parameters = dict(
        #
        #     # optimizer_parmas
        #     lr=[.01, .001],
        #
        #     # dataloader_parmas
        #     batch_size=[500, 1000],
        #     shuffle=[True, False],
        #
        #     # loss_function_parmas
        #     class_weight=[None, torch.tensor([0.8, 1])],
        #
        #     # model_parmas
        #     dropout_percent = [0.4, 0.5],
        #     list_layers_input_size=[[200, 100, 50], [200, 50]]
        # )

        parameters = dict(

            # optimizer_parmas
            lr=[.01, .001],

            # dataloader_parmas
            batch_size=[1000],
            shuffle=[True, False],

            # loss_function_parmas
            class_weight=[None],

            # model_parmas
            dropout_percent = [0.4],
            list_layers_input_size=[[200, 100, 50]]
        )

        self.list_all_combinations = list(self.product_dict(**parameters))

        self.__df_all_combinations = pd.DataFrame(self.list_all_combinations)
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

    def __extract_parmas(self, dict_parmas):

        # optimizer_parmas
        self.lr = dict_parmas['lr']
        # loss_function_parmas
        self.class_weight = dict_parmas['class_weight']
        # dataloader_parmas
        self.batch_size = dict_parmas['batch_size']
        self.shuffle = dict_parmas['shuffle']
        # model_parmas
        self.list_layers_input_size = dict_parmas['list_layers_input_size']
        self.dropout_percent = dict_parmas['dropout_percent']

    def __load_parmas(self):

        self.nn_model = NNModel(embedding_size=self._NNDataP.list_categorical_embed_sizes,
                                num_numerical_cols=len(self._NNDataP.list_col_numerical),
                                output_size=2,
                                list_layers_input_size=self.list_layers_input_size,
                                dropout_percent=self.dropout_percent)

        self.loss_function = nn.CrossEntropyLoss(weight=self.class_weight)
        
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=self.lr)

        self._NNDataP._create_dataloader(batch_size=self.batch_size, shuffle=self.shuffle)

    def __prepare_model_desc(self):
        """mark down the detail time and parmas"""

        self.str_ymd_hms = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

        # __________all parmas__________
        self.str_parmas_desc = ''

        # optimizer_parmas
        self.str_parmas_desc += f'_lr_{self.lr}'

        # model_parmas
        self.str_parmas_desc += f'_dropout_p_{self.dropout_percent}'
        # [1,2,3,4] --> '[1,2,3,4]'
        self.str_parmas_desc += f"_layers_size_[{','.join(str(size) for size in self.list_layers_input_size)}]"

        # dataloader_parmas
        self.str_parmas_desc += f'_bs_{self.batch_size}'
        self.str_parmas_desc += f'_shuffle_{self.shuffle}'

        # loss_function_parmas
        if self.class_weight is not None:
            self.str_parmas_desc += f'_cw_{self.class_weight[0]:.2f}_{self.class_weight[1]:.2f}'
        else:
            self.str_parmas_desc += '_no_cw'

    def __create_tsboard_writer(self):

        if self.log_dir is not None:

            self.writer = SummaryWriter(f'{self.log_dir}/{self.str_ymd_hms}{self.str_parmas_desc}')
        else:
            self.writer = SummaryWriter()

    def find_best_parmas_by_valid_loss(self, num_epochs=10, patience=10, is_refit=True):
        """is_refit: Refit an estimator using the best found parameters on the whole dataset.
        the best parameters is considered by the best valid loss"""

        for model_idx, dict_parmas in enumerate(self.list_all_combinations):

            self.__extract_parmas(dict_parmas)
            self.__load_parmas()

            self.__prepare_model_desc()
            self.__create_tsboard_writer()

            self.num_epochs = num_epochs
            self.patience = patience
            early_stopping = EarlyStopping(patience=self.patience, verbose=True)

            self.nn_model.parmas_desc = self.str_parmas_desc
            print('train and valid model: ', self.nn_model.parmas_desc)

            self.list_train_acc = []
            self.list_train_loss = []
            self.list_valid_acc = []
            self.list_valid_loss = []

            self.total_step = len(self._NNDataP.train_loader)

            for epoch in range(1, self.num_epochs + 1):

                # __________train and log___________
                running_loss_train = 0.0
                # scheduler.step(epoch)
                num_correct_train = 0
                num_total_train = 0

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

                    running_loss_train += single_loss.item()
                    _, y_pred_label = torch.max(y_pred, dim=1)
                    num_correct_train += torch.sum(y_pred_label == ts_y).item()
                    num_total_train += ts_y.size(0)
                    if self.is_display_batch_info:
                        if (batch_idx) % 5 == 0:
                            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                                  .format(epoch, self.num_epochs, batch_idx, self.total_step, single_loss.item()))

                train_accuracy = 100 * num_correct_train / num_total_train
                train_loss = running_loss_train / self.total_step

                if self.is_display_detail:
                    print(f'train loss: {train_loss:.4f}, train acc: {train_accuracy:.4f}')

                if self.log_dir is not None:
                    self.writer.add_scalar(
                        'Loss/Train', train_loss, epoch
                    )
                    self.writer.add_scalar(
                        'Accuracy/Train', train_accuracy, epoch
                    )

                self.list_train_acc.append(train_accuracy)
                self.list_train_loss.append(train_loss)

                # __________validation and log__________

                running_loss_valid = 0.0
                num_correct_valid = 0
                num_total_valid = 0

                # reduce memory consumption
                with torch.no_grad():
                    self.nn_model.eval()

                    for batch_idx, (ts_x_categ, ts_x_numer, ts_y) in enumerate(self._NNDataP.valid_loader):

                        y_pred = self.nn_model(ts_x_categ, ts_x_numer)
                        single_loss = self.loss_function(y_pred, ts_y)

                        running_loss_valid += single_loss.item()
                        _, y_pred_label = torch.max(y_pred, dim=1)
                        num_correct_valid += torch.sum(y_pred_label == ts_y).item()
                        num_total_valid += ts_y.size(0)

                    valid_accuracy = 100 * num_correct_valid / num_total_valid
                    valid_loss = running_loss_valid / len(self._NNDataP.valid_loader)

                    if self.is_display_detail:
                        print(f'valid loss: {valid_loss:.4f}, valid acc: {valid_accuracy:.4f}')

                    if self.log_dir is not None:
                        self.writer.add_scalar(
                            'Loss/Valid', valid_loss, epoch
                        )
                        self.writer.add_scalar(
                            'Accuracy/Valid', valid_accuracy, epoch
                        )

                    self.list_valid_acc.append(valid_accuracy)
                    self.list_valid_loss.append(valid_loss)

                # early_stopping needs the validation loss to check if it has improved,
                # and if it has, it will make a checkpoint of the current model
                early_stopping(valid_loss, self.nn_model)

                if early_stopping.early_stop:
                    print(f"Early stopping at {epoch}")
                    break

                if self.is_display_detail or self.is_display_batch_info:
                    print('')

            # load the last checkpoint with the best model
            self.nn_model.load_state_dict(torch.load('checkpoint.pt'))
            # back up all the model
            self.list_train_valid_model.append(self.nn_model)

        self.__prepare_validation_performance()
        if is_refit:
            for model in self.list_train_valid_model:
                if model.parmas_desc == self.df_validation_performance.loc[0, 'parmas_desc']:
                    self.best_model = model

            # TODO refit

    def __prepare_validation_performance(self):

        list_parmas_desc = []
        list_best_valid_loss = []

        for model in self.list_train_valid_model:
            list_parmas_desc.append(model.parmas_desc)
            list_best_valid_loss.append(model.best_valid_loss)

        df_best_valid_loss = pd.DataFrame({'parmas_desc': list_parmas_desc, 'best_valid_loss': list_best_valid_loss})

        self.df_validation_performance = pd.merge(self.__df_all_combinations, df_best_valid_loss,
                                                  left_index=True, right_index=True).sort_values('best_valid_loss')

    def test_model(self):

        self.list_test_acc = []
        self.list_test_loss = []
        self.ts_test_pred_label = torch.empty(0)

        running_loss_test = 0.0
        num_correct_test = 0
        num_total_test = 0

        # reduce memory consumption
        with torch.no_grad():
            self.nn_model.eval()

            for batch_idx, (ts_x_categ, ts_x_numer, ts_y) in enumerate(self._NNDataP.test_loader):

                y_pred = self.nn_model(ts_x_categ, ts_x_numer)
                single_loss = self.loss_function(y_pred, ts_y)

                running_loss_test += single_loss.item()
                _, y_pred_label = torch.max(y_pred, dim=1)
                self.ts_test_pred_label = torch.cat((self.ts_test_pred_label, y_pred_label))
                num_correct_test += torch.sum(y_pred_label == ts_y).item()
                num_total_test += ts_y.size(0)

            test_accuracy = 100 * num_correct_test / num_total_test
            test_loss = running_loss_test / len(self._NNDataP.test_loader)

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