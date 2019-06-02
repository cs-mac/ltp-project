"""
This baseline model is based on:
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging
"""

import unicodedata

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

torch.manual_seed(673)


# Load data
training_data = [
    ('The dog ate the apple'.split(), ['DET', 'NN', 'V', 'DET', 'NN']),
    ('Everybody read that book'.split(), ['NN', 'V', 'DET', 'NN'])
]


def create_value_map(vals):
    return {val: ix for ix, val in enumerate(vals)}


word_set = list(set([word for sent, _ in training_data for word in sent]))
word_to_ix = create_value_map(word_set)

char_set = ['<PAD>'] + [chr(i) for i in range(128)]
char_to_ix = create_value_map(char_set)

word_ix_to_char_x = {}

tag_set = ['DET', 'NN', 'V']
tag_to_ix = create_value_map(tag_set)


def to_tensor(seq, to_ix):
    return torch.tensor([to_ix[i] for i in seq], dtype=torch.long)


def to_word_tensor(words):
    return to_tensor(words, word_to_ix)


def to_tag_tensor(tags):
    return to_tensor(tags, tag_to_ix)


def to_char_tensor(word_ix):
    if word_ix not in word_ix_to_char_x:
        chars = unicodedata.normalize('NFKD', word_set[word_ix])
        word_ix_to_char_x[word_ix] = to_tensor(chars, char_to_ix)

    return word_ix_to_char_x[word_ix]


# Define model
class BiLSTMTagger(nn.Module):
    # https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#sphx-glr-beginner-nlp-advanced-tutorial-py
    # https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e

    def __init__(self, w_in_dim, c_in_dim, out_dim, w_emb_dim=128, c_emb_dim=100, hidden_dim=100, c_hidden_dim=100):
        super(BiLSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim

        self.c_emb_dim = c_emb_dim
        self.c_hidden_dim = c_hidden_dim

        # Word embedding
        self.word_embeds = nn.Embedding(w_in_dim, w_emb_dim)

        # Char embedding
        self.char_embeds = nn.Embedding(c_in_dim, c_emb_dim)
        self.char_bilstm = nn.LSTM(c_emb_dim, c_hidden_dim, num_layers=1,
                                   batch_first=True, bidirectional=True)

        # Network
        self.bilstm = nn.LSTM(w_emb_dim + 2 * c_emb_dim, hidden_dim, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_dim * 2, out_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def _embed_chars(self, word_iX):  # (batch_size, num_words) -> (batch_size, num_words, char_emb_dim)
        # Get character identities for each word
        char_X = [to_char_tensor(word_ix) for word_ix in word_iX.flatten()]

        # Zero pad char sequences to get equal lengths
        char_x_lengths = torch.tensor([len(x) for x in char_X])
        char_X = pad_sequence(char_X, batch_first=True)

        char_X = self.char_embeds(char_X)
        char_X = pack_padded_sequence(char_X, char_x_lengths, batch_first=True, enforce_sorted=False)
        _, (char_X, _) = self.char_bilstm(char_X)
        char_X = char_X.view(*word_iX.size(), 2 * self.c_emb_dim)

        return char_X

    def forward(self, word_iX):
        word_X = self.word_embeds(word_iX)
        char_X = self._embed_chars(word_iX)

        assert len(word_X.size()) == 3
        assert len(char_X.size()) == 3
        assert word_X.size()[:2] == char_X.size()[:2]

        X = torch.cat((word_X, char_X), dim=2)
        X, _ = self.bilstm(X)
        X = self.out(X)

        return self.softmax(X)


# Train model
model = BiLSTMTagger(len(word_set), len(char_set), len(tag_set))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epoch_losses = []
for epoch in range(300):
    epoch_loss = 0

    for words, tags in training_data:
        model.zero_grad()

        # Prepare single sample batch
        # TODO: Get data with DataLoader
        # TODO: Support variable length sentences with padding
        X_train = to_word_tensor(words).unsqueeze(0)
        y_train = to_tag_tensor(tags).unsqueeze(0)

        # Predict output
        y_pred = model(X_train)
        loss = loss_function(y_pred.view(-1, len(tag_set)), y_train.view(-1))
        epoch_loss += loss.item()

        # Backpropagate
        loss.backward()
        optimizer.step()

    epoch_losses.append(epoch_loss)
    print('Epoch {} loss: {}'.format(epoch, epoch_loss))


# Evaluate results
with torch.no_grad():
    for words, tags in training_data:
        X_train = to_tensor(words, word_to_ix).unsqueeze(0)
        y_pred = model(X_train)

        print('Sentence: {}'.format(words))
        print('Target tags:    {}'.format(tags))
        print('Predicted tags: {}'.format([tag_set[i] for i in torch.argmax(y_pred[0], dim=1)]))
        print('')
