from dataloader import get_data, random_training_example
from preprocess import n_letters
from classifier import RNNModel
from evaluation import category_from_output, time_since

import time
learning_rate = 0.005

def main():
    path = 'data/names/*.txt'
    category_lines, all_categories = get_data(path)

    hidden_size = 128
    rnn = RNNModel(input_size=n_letters, hidden_size=hidden_size, output_size=len(all_categories), rnn_type="GRU")

    n_iters = 100000
    print_every = 5000
    current_loss = 0
    start = time.time()
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)
        output, loss = train(rnn, line_tensor, category_tensor)
        current_loss += loss

        if iter % print_every == 0:
            guess, guess_i = category_from_output(output, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
            iter, iter / n_iters * 100, time_since(start), loss, line, guess, correct))


def train(rnn, line_tensor, category_tensor):
    rnn.zero_grad()

    pred = rnn(line_tensor)

    loss = rnn.criterion(pred, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return pred, loss.item()

if __name__ == "__main__":
    main()
