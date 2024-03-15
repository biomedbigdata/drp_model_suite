"""
Code for the MOLI model.
Original authors: Sharifi-Noghabi et al. (2019, 10.1093/bioinformatics/btz318)
Code adapted from: Hauptmann et al. (2023, 10.1186/s12859-023-05166-7), https://github.com/kramerlab/Multi-Omics_analysis
"""
import torch
from torch import nn


class MOLIEncoder(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(MOLIEncoder, self).__init__()
        self.encode = torch.nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.encode(x)


class MOLIClassifier(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super(MOLIClassifier, self).__init__()
        self.classifier = torch.nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, 1),
        )

    def forward(self, x):
        return self.classifier(x)


class Moli(nn.Module):
    def __init__(self, input_sizes, output_sizes, dropout_rates):
        super(Moli, self).__init__()
        self.expression_encoder = MOLIEncoder(input_sizes[0], output_sizes[0], dropout_rates[0])
        self.mutation_encoder = MOLIEncoder(input_sizes[1], output_sizes[1], dropout_rates[1])
        self.cna_encoder = MOLIEncoder(input_sizes[2], output_sizes[2], dropout_rates[2])
        self.classifier = MOLIClassifier(output_sizes[0] + output_sizes[1] + output_sizes[2], dropout_rates[3])

    def forward_with_features(self, expression, mutation, cna):
        left_out = self.expression_encoder(expression)
        middle_out = self.mutation_encoder(mutation)
        right_out = self.cna_encoder(cna)
        left_middle_right = torch.cat((left_out, middle_out, right_out), 1)
        return [self.classifier(left_middle_right), left_middle_right]

    def forward(self, expression, mutation, cna):
        left_out = self.expression_encoder(expression)
        middle_out = self.mutation_encoder(mutation)
        right_out = self.cna_encoder(cna)
        left_middle_right = torch.cat((left_out, middle_out, right_out), 1)
        return self.classifier(left_middle_right)