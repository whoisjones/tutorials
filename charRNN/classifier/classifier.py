import torch
from torch.nn import RNN, LSTM, GRU


class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_type: str = "RNN"):
        super(RNNModel, self).__init__()

        if rnn_type == "RNN":
            self.rnn_layer = RNN(input_size=input_size, hidden_size=hidden_size)
        elif rnn_type == "LSTM":
            self.rnn_layer = LSTM(input_size=input_size, hidden_size=hidden_size)
        else:
            self.rnn_layer = GRU(input_size=input_size, hidden_size=hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.criterion = torch.nn.NLLLoss()

    def forward(self, input):
        output, hn = self.rnn_layer(input)
        logits = self.linear(output[-1])
        pred = self.softmax(logits)
        return pred

    def predict(self, output):
        return self.softmax(output)

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
