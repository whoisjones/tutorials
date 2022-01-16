import torch


class Generator(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.rnn = torch.nn.RNN(input_size=input_size, output_size=output_size)

    def forward(self):
        pass