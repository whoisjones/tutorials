import torch


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.input2hidden = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.input2output = torch.nn.Linear(input_size + hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.criterion = torch.nn.NLLLoss()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.input2hidden(combined)
        output = self.input2output(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
