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
import utils

from tqdm import tqdm

def train(model, optimizer, data_iter, text_field, args):
	model.train()
	loss_function = nn.NLLLoss()
	avg_loss = 0
	total = 0
	iter_len = len(data_iter)
	batch_idx = 0
	batches = [batch for batch in data_iter]

	for idx in range(len(batches)):
		# TODO: fix API based on torchtext output; make sure they are Variables
		batch = batches[idx]
		if idx % 100 == 0:
			print("finish %d" % idx)
		#print("assign batch")
		context = torch.transpose(batch.text, 0, 1)
		target = batch.target[-1,:] 
		batch_size = context.size(0)

		optimizer.zero_grad()
		output = model(context)

		loss = loss_function(output, target)
		avg_loss += loss.data.numpy()[0] * batch_size
		total += batch_size

		loss.backward() 
		optimizer.step()
		if batch_idx >= iter_len - 2:
			break

		batch_idx += 1

	avg_loss /= total
	return model, optimizer, np.exp(avg_loss)

def evaluate(model, data_iter, text_field, args):
	model.eval()
	loss_function = nn.NLLLoss()
	avg_loss = 0
	total = 0
	iter_len = len(data_iter)
	batch_idx = 0


	batches = [batch for batch in data_iter]
	for idx in range(len(batches)):
		if idx % 100 == 0:
			print("finish %d" % idx)
		batch = batches[idx]
		# TODO: fix API based on torchtext output; make sure they are Variables
		context = torch.transpose(batch.text, 0, 1)

		target = batch.target[-1,:] 
		batch_size = context.size(0)

		output = model(context)
		loss = loss_function(output, target)
		avg_loss += loss.data.numpy()[0] * batch_size
		total += batch_size

		if batch_idx >= iter_len - 2:
			break

		batch_idx += 1
	avg_loss /=  total

	return np.exp(avg_loss)

def main():
	# TODO: change this API according to utils.py implementation
	train_iter, val_iter, test_iter, text_field = utils.load_ptb(ptb_path='data.zip', 
									ptb_dir='data', bptt_len=args.context_size, 
									batch_size=args.batch_size, gpu=args.GPU, 
        							reuse=False, repeat=False, shuffle=True)

	model = LBL(text_field.vocab.vectors, args.context_size)	
	optimizer = optim.Adam(model.get_train_parameters(), lr=args.lr)
	for epoch in range(args.epochs):
		# TODO: do we need to return model?
		print("epoch begin %d" % epoch)
		model, optimizer, avg_loss = train(model, optimizer, train_iter, text_field, args)
		# TODO: evaluate
		print("TRAIN [EPOCH %d]: PERPLEXITY %.5lf" % (epoch, avg_loss))
		if epoch % 1 == 0:
			val_perp  = evaluate(model, val_iter, text_field, args)
			test_perp = evaluate(model, test_iter, text_field, args)
			print("VALIDATE [EPOCH %d]: PERPLEXITY %.5lf" % (epoch, val_perp))
			print("TEST [EPOCH %d]: PERPLEXITY %.5lf" % (epoch, test_perp))

if __name__ == "__main__":
	main()