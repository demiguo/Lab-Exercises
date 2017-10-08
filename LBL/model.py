import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import re

class LBL(nn.Module):
    def __init__(self, pretrained_embeds, context_size):
        super(LBL, self).__init__()

        self.context_size = context_size # n in the paper
        self.hidden_size = pretrained_embeds.size(1)
        self.vocab_size = pretrained_embeds.size(0)

        self.embedding_layer = nn.Embedding(self.vocab_size, self.hidden_size)
        self.context_layer = nn.Linear(self.hidden_size * self.context_size, self.hidden_size, bias=False) # C in the paper
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size) # dot product + bias in the paper

        self.init_weight(pretrained_embeds) # pretrained embeds is R in the paper

    def get_train_parameters(self):
        params = []
        for param in self.parameters():
            if param.requires_grad == True:
                params.append(param)
        return params

    def init_weight(self,pretrained_embeds):
        assert(pretrained_embeds.size() == (self.vocab_size, self.hidden_size))
        self.embedding_layer.weight.data.copy_(pretrained_embeds)
        self.output_layer.weight.data = self.embedding_layer.weight.data

    def forward(self, context_words):
        self.batch_size = context_words.size(0)
        assert context_words.size(1) == self.context_size

        embeddings = self.embedding_layer(context_words)
        assert embeddings.size() == (self.batch_size, self.context_size, self.hidden_size) # sanity check
        context_vectors = self.context_layer(embeddings.view(self.batch_size, self.context_size * self.hidden_size))
        assert context_vectors.size() == (self.batch_size, self.hidden_size)
        raw_outputs = self.output_layer(context_vectors)
        assert raw_outputs.size() == (self.batch_size, self.vocab_size)
        outputs = F.log_softmax(raw_outputs)
        assert outputs.size() == (self.batch_size, self.vocab_size)
        return outputs
