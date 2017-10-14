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
	loss_function_tot = nn.NLLLoss(size_average=False)
	loss_function_avg = nn.NLLLoss(size_average=True)
	total_loss = 0
	data_size = 0
	
	iter_len = len(data_iter)
	batch_idx = 0
	for batch in data_iter:
		context = torch.transpose(batch.text, 0, 1)
		target = batch.target[-1,:] 
		batch_size = context.size(0)
		# zero out gradients
		optimizer.zero_grad()
		# get output
		output = model(context)
		# calculate loss
		loss = loss_function_avg(output, target)
		total_loss += loss_function_tot(output, target).data.numpy()[0]
		data_size += batch_size
		# calculate gradients
		loss.backward() 
		# update parameters
		optimizer.step()

		# skip the last batch
		if batch_idx >= iter_len - 2:
			break

		batch_idx += 1

	avg_loss = total_loss / data_size
	return model, optimizer, np.exp(avg_loss)

def evaluate(model, data_iter, text_field, args):
	model.eval()
	loss_function_tot = nn.NLLLoss(size_average=False)
	total_loss = 0
	data_size = 0
	iter_len = len(data_iter)
	batch_idx = 0
	for batch in data_iter:
		context = torch.transpose(batch.text, 0, 1)
		target = batch.target[-1,:] 
		batch_size = context.size(0)
		# get model output
		output = model(context)
		# calculate total loss
		loss = loss_function_tot(output, target) # loss is already averaged
		total_loss += loss.data.numpy()[0]
		data_size += batch_size

		# skip last batch
		if batch_idx >= iter_len - 2:
			break

		batch_idx += 1

	avg_loss = total_loss / data_size
	perplexity = np.exp(avg_loss) # use exp here because the loss uses ln 
	return perplexity

def main():
	train_iter, val_iter, test_iter, text_field = utils.load_ptb(ptb_path='data.zip', 
									ptb_dir='data', bptt_len=args.context_size, 
									batch_size=args.batch_size, gpu=args.GPU, 
        							reuse=False, repeat=False, shuffle=True)
	lr = args.initial_lr
	model = LBL(text_field.vocab.vectors, args.context_size, args.dropout)	
	print("Model: %s" % model)
	# optimizer = optim.SGD(model.get_train_parameters(), lr=args.lr)
	if args.optimizer == "Adamax":
		print("Optimizer: Adamax")
		optimizer = optim.Adamax(model.get_train_parameters(), lr=lr, weight_decay=1e-5)
	elif args.optimizer == "Adam":
		print("Optimizer: Adam")
		optimizer = optim.Adam(model.get_train_parameters(), lr=lr, weight_decay=1e-5)
	elif args.optimizer == "SGD":
		print("Optimizer: SGD")
		optimizer = optim.SGD(model.get_train_parameters(), lr=lr, weight_decay=1e-5)
	else:
		assert False, "Optimizer %s not found" % args.optimizer

	val_perps = []
	for epoch in range(args.epochs):
		model, optimizer, train_perp = train(model, optimizer, train_iter, text_field, args)
		print("TRAIN [EPOCH %d]: PERPLEXITY %.5lf" % (epoch, train_perp))
		val_perp  = evaluate(model, val_iter, text_field, args)
		print("VALIDATE [EPOCH %d]: PERPLEXITY %.5lf" % (epoch, val_perp))
		val_perps.append(val_perp)

		# adjust leraning rate
		if len(val_perps) > args.adapt_lr_epoch and np.min(val_perps[-args.adapt_lr_epoch:]) > np.min(val_perps[:-args.adapt_lr_epoch]):
			lr = lr * 0.5
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr 

		if epoch % 5 == 0:
			test_perp = evaluate(model, test_iter, text_field, args)
			print("TEST [EPOCH %d]: PERPLEXITY %.5lf" % (epoch, test_perp))

if __name__ == "__main__":
	main()