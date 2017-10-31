import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class LBL(nn.Module):
    def __init__(self, pretrained_embeds, context_size, dropout=0.):
        super(LBL, self).__init__()
        # n in the paper
        self.context_size = context_size
        self.hidden_size = pretrained_embeds.size(1)
        self.vocab_size = pretrained_embeds.size(0)

        self.embedding_layer = nn.Embedding(
                self.vocab_size, self.hidden_size)
        self.max_norm_embedding()
        # C in the paper
        self.context_layer = nn.Linear(
                self.hidden_size * self.context_size,
                self.hidden_size, bias=False)
        # dot product + bias in the paper
        self.output_layer =\
            nn.Linear(self.hidden_size, self.vocab_size)

        self.dropout = nn.Dropout(p=dropout)

    def get_train_parameters(self):
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def max_norm_embedding(self, max_norm=1):
        norms = torch.norm(self.embedding_layer.weight, p=2, dim=1)
        to_rescale = Variable(torch.from_numpy(
                np.where(norms.data.numpy() > max_norm)[0]))
        norms = torch.norm(self.embedding_layer(to_rescale), p=2, dim=1).data
        scaled = self.embedding_layer(to_rescale).div(
                Variable(norms.view(len(to_rescale), 1).expand_as(
                        self.embedding_layer(to_rescale)))).data
        self.embedding_layer.weight.data[to_rescale.long().data] = scaled

    def forward(self, context_words):
        self.batch_size = context_words.size(0)
        assert context_words.size(1) == self.context_size, \
            "context_words.size()=%s | context_size=%d" % \
            (context_words.size(), self.context_size)

        embeddings = self.embedding_layer(context_words)
        # sanity check
        assert embeddings.size() == \
            (self.batch_size, self.context_size, self.hidden_size)
        context_vectors = self.context_layer(embeddings.view(
                self.batch_size, self.context_size * self.hidden_size))
        context_vectors = self.dropout(context_vectors)
        assert context_vectors.size() == (self.batch_size, self.hidden_size)
        raw_outputs = self.output_layer(context_vectors)
        assert raw_outputs.size() == (self.batch_size, self.vocab_size)
        outputs = F.log_softmax(raw_outputs)
        assert outputs.size() == (self.batch_size, self.vocab_size)
        return outputs
