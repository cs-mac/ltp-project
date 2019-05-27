"""
This baseline model is based on:
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(673)


# Load data
training_data = [
    ('The dog ate the apple'.split(), ['DET', 'NN', 'V', 'DET', 'NN']),
    ('Everybody read that book'.split(), ['NN', 'V', 'DET', 'NN'])
]


# Prepare data
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


vocab = list(set([word for sent, _ in training_data for word in sent]))
word_to_ix = {word: ix for ix, word in enumerate(vocab)}

tagset = ['DET', 'NN', 'V']
tag_to_ix = {tag: ix for ix, tag in enumerate(tagset)}


# Define model
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sent):
        x = self.embeddings(sent)
        x, _ = self.lstm(x.view(len(sent), 1, -1))
        x = self.linear1(x.view(len(sent), -1))
        scores = F.log_softmax(x, dim=1)
        return scores


# Train model
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(vocab), len(tagset))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(300):
    for sent, tags in training_data:
        model.zero_grad()

        sent_in = prepare_sequence(sent, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        tag_scores = model(sent_in)
        loss = loss_function(tag_scores, targets)

        loss.backward()
        optimizer.step()


# Evaluate results
with torch.no_grad():
    for sent, tags in training_data:
        sent_in = prepare_sequence(sent, word_to_ix)
        tag_scores = model(sent_in)

        print('Sentence: {}'.format(sent))
        print('Target tags:    {}'.format(tags))
        print('Predicted tags: {}'.format([tagset[i] for i in torch.argmax(tag_scores, 1)]))
        print('')
