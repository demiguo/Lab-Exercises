from __future__ import print_function

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import re
from matplotlib import pyplot as plt

from torchtext import data 
from torchtext import datasets

from config import args

def main():
	# TODO: change this API according to utils.py implementation
	train_iter, val_iter, test_iter, text_field = utils.preprocess(args)
	

if __name__ == "__main__":
	main()