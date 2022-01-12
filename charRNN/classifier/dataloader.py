import os
import glob
import random
from io import open

import torch
from preprocess import unicode_to_ascii, line_to_tensor


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
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor
