import torch
import numpy as np


class ParmasSelector(object):

    def select(self, str_selection):

        method_name = f'get_parmas_{str_selection}'
        method = getattr(self, method_name)
        return method()

    def show_available_parmas_options(self):
        return [selection.replace('get_parmas_', '') for selection in dir(self) if
                'get_parmas_' in selection]

    @staticmethod
    def get_dict_parmas_to_abbrev():
        """if the parmas can be found in this dict,
        tsboard and epoch performance will be logged by their abbreviation"""

        return {
            'optimizer_attr': 'optim',
            'amsgrad': 'ams',
            'batch_size': 'bs',
            'shuffle': 'sf',
            'class_weight': 'cw',
            'dropout_percent': 'dp',
            'list_layers_input_size': 'ls',
            'oversampling_w': 'ow'
        }

    @staticmethod
    def get_parmas_baseline():

        return dict(

            # optimizer_parmas
            optimizer_attr=['Adam'],
            lr=[0.02],
            amsgrad=[False],

            # dataloader_parmas
            batch_size=[1000],
            shuffle=[True, False],

            # loss_function_parmas
            # e.g. None or torch.tensor([x.0, y.0])
            class_weight=[None],

            # model_parmas
            dropout_percent=[0.4, 0.5],
            list_layers_input_size=[[200, 100, 50]],

            # oversampling_weight
            # e.g. 'count_balance' or np.array([x, y])
            oversampling_w=[None]
        )

    @staticmethod
    def get_parmas_with_class_weight():

        return dict(

            # optimizer_parmas
            optimizer_attr=['Adam'],
            lr=[0.02],
            amsgrad=[False],

            # dataloader_parmas
            batch_size=[1000],
            shuffle=[True],

            # loss_function_parmas
            # e.g. None or torch.tensor([x.0, y.0])
            class_weight=[torch.tensor([1.0, 4.0]), torch.tensor([1.0, 3.0]), torch.tensor([1.0, 2.0])],

            # model_parmas
            dropout_percent=[0.4, 0.5],
            list_layers_input_size=[[200, 100, 50]],

            # oversampling_weight
            # e.g. 'count_balance' or np.array([x, y])
            oversampling_w=[None]
        )

    @staticmethod
    def get_parmas_with_oversampling():

        return dict(

            # optimizer_parmas
            optimizer_attr=['Adam'],
            lr=[0.02],
            amsgrad=[False],

            # dataloader_parmas
            batch_size=[1000],
            shuffle=[False],

            # loss_function_parmas
            # e.g. None or torch.tensor([x.0, y.0])
            class_weight=[None],

            # model_parmas
            dropout_percent=[0.4, 0.5],
            list_layers_input_size=[[200, 100, 50]],

            # oversampling_weight
            # e.g. 'count_balance' or np.array([x, y])
            oversampling_w=[np.array([1, 4]), np.array([1, 3]), np.array([1, 2])]
        )
