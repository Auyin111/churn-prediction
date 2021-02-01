import torch
import torch.nn as nn
import datetime
import sklearn.metrics as metrics
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from lib.data_preprocessor import NNDataPreprocess
from lib.model import NNModel
from lib.early_stopping import EarlyStopping


class ChurnPrediction:

    def __init__(self, df_all_data, class_weight=None, is_display_detail=True, is_display_batch_info = False,
                 log_dir='C:/Users/Auyin/PycharmProjects/churn-prediction/train_valid_log',
                 ):

        self.class_weight = class_weight
        self.is_display_detail = is_display_detail
        self.is_display_batch_info = is_display_batch_info
        self.log_dir = log_dir


        self._NNDataP = NNDataPreprocess(df_all_data, test_fraction=0.2)

        self.nn_model = NNModel(embedding_size=self._NNDataP.list_categorical_embed_sizes,
                                num_numerical_cols=len(self._NNDataP.list_col_numerical),
                                output_size=2,
                                layers=[200, 100, 50], p=0.4)
        self.__clare_model_setting()

        # init variable
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

    def __clare_model_setting(self):

        if self.class_weight is not None:
            self.loss_function = nn.CrossEntropyLoss(weight=self.class_weight)
        else:
            self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)

    def preview_model(self):

        return print(self.nn_model)

    def train_valid_model(self, num_epochs=10, patience=10):

        self.num_epochs = num_epochs
        self.patience = patience
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        if self.log_dir is not None:
            if self.class_weight is not None:
                writer = SummaryWriter(
                    f"""{self.log_dir}/{datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}_with_class_weight""")
            else:
                writer = SummaryWriter(
                    f"""{self.log_dir}/{datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}""")

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
                writer.add_scalar(
                    'Loss/Train', train_loss, epoch
                )
                writer.add_scalar(
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
                    writer.add_scalar(
                        'Loss/Valid', valid_loss, epoch
                    )
                    writer.add_scalar(
                        'Accuracy/Valid', valid_accuracy, epoch
                    )

                self.list_valid_acc.append(valid_accuracy)
                self.list_valid_loss.append(valid_loss)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, self.nn_model)

            if early_stopping.early_stop:
                print(f"Early stopping at {epoch}")
                break

            if self.is_display_detail or self.is_display_batch_info:
                print('')

        # load the last checkpoint with the best model
        self.nn_model.load_state_dict(torch.load('checkpoint.pt'))

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