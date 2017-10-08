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
from model import LBL

def train(model, optimizer, data_iter, text_field, args):
	model.train()
	loss_function = nn.NLLLoss()
	avg_loss = 0
	total = 0
	for batch_idx, batch in enumerate(data_iter):
		# TODO: fix API based on torchtext output; make sure they are Variables
		context = torch.transpose(batch.text, (0,1))
		target = batch.target[-1,:] 
		batch_size = context.size(0)
		if batch_idx == 0:
			print "batch_size=", batch_size, "context.size()=", context.size(), " target.size()=", target.size()

		optimizer.zero_grad()
		output = model(x)

		loss = loss_function(output, target)
		avg_loss += loss.data.numpy()[0]
		loss /= batch_size
		total += batch_size
		
		loss.backward() # TODO: do i first get the avg loss?
		optimizer.step()

	avg_loss /= 1.0 * batch_size
	return model, optimizer, avg_loss

def evaluate(model, data_iter, text_field, args):
	model.eval()
	print "not implemented"

def main():
	# TODO: change this API according to utils.py implementation
	train_iter, val_iter, test_iter, text_field = utils.load_ptb(args)

	model = LBL(text_field.vectors, args)	
	optimizer = optim.SGD(model.get_train_parameters(), lr=args.lr)
	for epoch in range(args.epoch):
		# TODO: do we need to return model?
		model, optimizer, avg_loss = train(model, optimizer, train_iter, text_field, args)
		# TODO: evaluate


if __name__ == "__main__":
	main()