
class LBL(nn.Module):

    # R, C, b
    def __init__(self, vocab_size, hid_layer_size, context_size, R):
        super(LBL, self).__init__()
        # init configuration
        self.vocab_size = vocab_size
        self.context_size = context_size
        print("vocab_size=", vocab_size)
        print("R.shape=", R.size())
        self.hid_layer_size = hid_layer_size
        # embedding layers
        self.word_embeds = nn.Embedding(vocab_size, hid_layer_size)
        # Weight matrix, d to hidden layer
        self.C = nn.Linear(hid_layer_size * context_size, hid_layer_size, bias=False)
        # Bias in softmax layer
        self.bias = nn.Parameter(torch.ones(vocab_size)).view(self.vocab_size, 1)

        self.init_weight(R)
        self.R = autograd.Variable(R)

    def get_train_parameters(self):
        params = []
        for param in self.parameters():
            if param.requires_grad == True:
                params.append(param)
        return params

    def init_weight(self, glove_weight):
        assert(glove_weight.size() == (self.vocab_size, self.hid_layer_size))
        self.word_embeds.weight.data.copy_(glove_weight)
        self.word_embeds.weight.requires_grad = False

    def forward(self, context_vect):
        context_vect = self.word_embeds(context_vect)
        context_vect = context_vect.view(1, self.context_size * self.hid_layer_size)
        model_vect = self.C(context_vect).view(self.hid_layer_size, 1)
        final_vect = torch.mm(self.R, model_vect) + self.bias
        final_vect = F.log_softmax(final_vect.view(self.vocab_size))
        final_vect = final_vect.view(1, self.vocab_size)
        return final_vect

