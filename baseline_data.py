"""
This baseline model is based on:
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

torch.manual_seed(673)

def parse_arguments():
    parser = argparse.ArgumentParser(description='POS-tags using neural network')
    parser.add_argument('--input', metavar='FILE', help='File containing UD dependencies')
    return parser.parse_args()

def data_maker(file):
    file = open(file, "r")
    training_data = []
    for line in file:
        line.strip()
        if line[0] != "#" and line[0] != "":
            line.strip()
            line = line.strip("\n").split("\t")
            index = 0
            if len(line) > 1:
                print(int(line[0]))
            # if len(line) > 1:
            #     word = line[1]
            #     tag = line[3]
            #     training_data.append((word,tag))

# Prepare data
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


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

def main():
    # Load data
    training_data = [
        ('The dog ate the apple'.split(), ['DET', 'NN', 'V', 'DET', 'NN']),
        ('Everybody read that book'.split(), ['NN', 'V', 'DET', 'NN'])
    ]

    vocab = list(set([word for sent, _ in training_data for word in sent]))
    word_to_ix = {word: ix for ix, word in enumerate(vocab)}

    tagset = ['DET', 'NN', 'V']
    tag_to_ix = {tag: ix for ix, tag in enumerate(tagset)}


    # Define model
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6


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

if __name__ == "__main__":
    args = parse_arguments()
    if not args.input:
        print("Please provide an input filename")
    else:
        data_maker(args.input)
        main()
