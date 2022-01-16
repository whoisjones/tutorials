import os
import glob
import random
from io import open

import torch
from preprocess import unicode_to_ascii, all_letters


def findFiles(path):
    return glob.glob(path)


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


def get_data(path: str):
    category_lines = {}
    all_categories = []
    for filename in findFiles(path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    return category_lines, all_categories


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_training_example(category_lines, all_categories):
    category, line = random_pair(category_lines, all_categories)
    category_tensor = get_category_tensor(category,all_categories,len(all_categories))
    input_line_tensor = get_input_tensor(line, all_letters, len(all_letters))
    target_line_tensor = get_target_tensor(line, )
    return category_tensor, input_line_tensor, target_line_tensor


def random_pair(category_lines, all_categories):
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    return category, line


def get_category_tensor(category, all_categories, n_categories):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


def get_input_tensor(line, all_letters, n_letters):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


def get_target_tensor(line, all_letters, n_letters):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)
