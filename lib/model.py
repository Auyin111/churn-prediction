import torch
import torch.nn as nn


class NNModel(nn.Module):

    def __init__(self, embedding_size, num_numerical_cols, output_size, list_layers_input_size, dropout_percent=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(dropout_percent)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols

        all_layers = []
        for i in list_layers_input_size:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(dropout_percent))
            input_size = i

        all_layers.append(nn.Linear(list_layers_input_size[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

        print("NNModel object created")

    def forward(self, x_categorical, x_numerical):
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)

        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)
        return x