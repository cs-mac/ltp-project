"""
This baseline model is based on:
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from conllu.parser import parse_line, DEFAULT_FIELDS
from typing import Iterator, List, Dict, Tuple, Iterable

from allennlp.modules.elmo import Elmo, batch_to_ids
from itertools import chain

torch.manual_seed(673)

def parse_arguments():
    parser = argparse.ArgumentParser(description='POS-tags using neural network')
    parser.add_argument('--input', metavar='FILE', help='File containing UD dependencies')
    return parser.parse_args()

def lazy_parse(text: str, fields: Tuple[str, ...]=DEFAULT_FIELDS):
    '''
    Parses a conllu file
    '''
    for sentence in text.split("\n\n"):
        if sentence:
            yield [parse_line(line, fields)
                   for line in sentence.split("\n")
                   if line and not line.strip().startswith("#")]

def data_maker(file_path: str):
    '''
    Returns a tuple containing the words with their corresponding POS-tags
    from the UD-dataset
    '''
    with open(file_path, 'r') as conllu_file:
        for annotation in  lazy_parse(conllu_file.read()):
            annotation = [x for x in annotation if x["id"] is not None]
            sentences = [x["form"] for x in annotation]
            pos_tags = [x["upostag"] for x in annotation]
            #print(f'Sentence =\n{words}\n{pos_tags}\n')
            #print(f'Sentence =\n{len(words)}\n{len(pos_tags)}\n')
            yield (sentences, pos_tags)


def data_completinator(file):
    '''
    Chain independent sentence/postag generators to one big generator
    for given input
    '''
    return chain(data_maker(file))


def to_elmo(sentences):
    '''
    Use elmo pre-trained word embeddings
    '''
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elmo = Elmo(options_file, weight_file, 2, dropout=0)

    sentences = [sentence for sentence in sentences]
    character_ids = batch_to_ids(sentences)

    embeddings = elmo(character_ids)
    return embeddings

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

def main(training_data, elmo_embeddings):
    print(training_data)
    # Load data
    # training_data = [
    #     ('The dog ate the apple'.split(), ['DET', 'NN', 'V', 'DET', 'NN']),
    #     ('Everybody read that book'.split(), ['NN', 'V', 'DET', 'NN'])
    # ]

    vocab = list(set([word for sent, _ in training_data for word in sent]))
    word_to_ix = {word: ix for ix, word in enumerate(vocab)}

    tagset = ['DET', 'NN', 'V']
    tagset = list(set([tag for _, tags in training_data for tag in tags]))
    tag_to_ix = {tag: ix for ix, tag in enumerate(tagset)}
    print(tagset)


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
        # data_maker(args.input)
        training_data = data_completinator(args.input)
        for tuple in training_data:
            print(tuple)
        #elmo_embeddings = ""# = to_elmo(training_data[0])
        #main(training_data, elmo_embeddings)
