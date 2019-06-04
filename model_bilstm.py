"""
This baseline model is based on:
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging
"""

import os
import pickle
import argparse

import unicodedata
import conllu

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from allennlp.modules.elmo import Elmo, batch_to_ids

torch.manual_seed(673)


ELMO_OPTIONS_FILE = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json'
ELMO_WEIGHT_FILE = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
ELMO_BATCH_SIZE = 64


def lazy_parse(text):
    '''
    Parses a conllu file
    '''
    for sentence in text.split("\n\n"):
        if sentence:
            yield [conllu.parser.parse_line(line)
                   for line in sentence.split("\n")
                   if line and not line.strip().startswith("#")]


def conllu_reader(file_path):
    '''
    Returns a tuple containing the words with their corresponding POS-tags
    from the UD-dataset
    '''
    with open(file_path, 'r') as conllu_file:
        for annotation in lazy_parse(conllu_file.read()):
            annotation = [x for x in annotation if x["id"] is not None]
            sentence = [x["form"] for x in annotation]
            pos_tags = [x["upostag"] for x in annotation]
            yield (sentence, pos_tags)


def data_maker_bert(data):
    '''
    Transform data using BERT Wordpieces, and [CLS], [SEP] tags
    '''    
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_basic_tokenize=False)
    
    for sentence, tags in data:
        tok_sentence = []
        corrected_tags = []
        for word, tag in zip(sentence, tags):
            tokenized_text = tokenizer.tokenize(word)
            if len(tokenized_text) > 1:
                tok_sentence.append([tokenized_text[0]])
                corrected_tags.append(tag)
                for i in range(1, len(tokenized_text)):
                    tok_sentence.append([tokenized_text[i]])
                    corrected_tags.append('<IGN>')
            else:
                tok_sentence.append(tokenized_text)
                corrected_tags.append(tag)
        yield (['[CLS]']+tok_sentence+['[SEP]'], ['<PAD>']+corrected_tags+['<PAD>'])


class UniversalDependenciesDataset(Dataset):
    def __init__(self, root_path, train_file, embeds=None, cache_data=False):
        self.sent_tokens = []
        self.sent_tags = []
        self.sent_embeds = None
        self.sent_lengths = None

        self.token_set = ['<UNK>']
        self.char_set = ['<PAD>', '<UNK>'] + [chr(i) for i in range(128)]
        self.tag_set = []

        if not self._load_cached_data(cache_data, train_file, embeds):
            self._read_conllu(os.path.join(root_path, train_file))
            self._prepare_embeds(embeds)

        self._cache_data(cache_data, train_file, embeds)

        self.token_to_ix = self._create_value_map(self.token_set)
        self.char_to_ix = self._create_value_map(self.char_set)
        self.tag_to_ix = self._create_value_map(self.tag_set)
        self.token_ix_to_char_x = {}

    def __len__(self):
        return len(self.sent_tokens)

    def __getitem__(self, idx):
        return self.sent_tokens[idx], self.sent_tags[idx]

    def get_word_tensor(self, sent_ix):
        return self._get_tensor(self.sent_tokens[sent_ix], self.token_to_ix, unk_ix=0)

    def get_word_embeds(self, sent_ix):
        if self.sent_embeds is None:
            return None

        return self.sent_embeds[sent_ix, :self.sent_lengths[sent_ix]]

    def get_embed_dim(self, default=None):
        if self.sent_embeds is None:
            return default
        return self.sent_embeds.size()[2]

    def get_char_tensor(self, token_ix):
        if token_ix not in self.token_ix_to_char_x:
            chars = unicodedata.normalize('NFKD', self.token_set[token_ix])
            self.token_ix_to_char_x[token_ix] = self._get_tensor(chars, self.char_to_ix, unk_ix=1)

        return self.token_ix_to_char_x[token_ix]

    def get_tag_tensor(self, sent_ix):
        return self._get_tensor(self.sent_tags[sent_ix], self.tag_to_ix)

    def _get_tensor(self, seq, to_ix, unk_ix=None):
        return torch.tensor([to_ix[i] if i in to_ix or unk_ix is None else unk_ix for i in seq], dtype=torch.long)

    def _create_value_map(self, vals):
        return {val: ix for ix, val in enumerate(vals)}

    def _read_conllu(self, conllu_path):
        for tokens, tags in conllu_reader(conllu_path):
            self.sent_tokens.append(tokens)
            self.sent_tags.append(tags)

            for token in tokens:
                if token not in self.token_set:
                    self.token_set.append(token)
            for tag in tags:
                if tag not in self.tag_set:
                    self.tag_set.append(tag)

    def _prepare_embeds(self, embeds):
        if embeds is None:
            return
        if embeds == 'elmo':
            return self._embed_elmo()
        if embeds == 'bert':
            return self._embed_bert()
        print('Unknown embedding type: {}'.format(embeds))

    def _embed_elmo(self):
        print(':: Initializing ELMo')
        elmo = Elmo(ELMO_OPTIONS_FILE, ELMO_WEIGHT_FILE, 1, dropout=0)

        print(':: Encoding tokens')
        char_ids = batch_to_ids(self.sent_tokens)

        print(':: Calculating ELMo embeddings')
        all_sent_embeds = []
        all_sent_lengths = []
        for i in range(0, len(char_ids), ELMO_BATCH_SIZE):
            print(' > {}/{} [{}%]'.format(i, len(char_ids), int(i / len(char_ids) * 100)), end='\r')

            elmo_batch = char_ids[i:i+ELMO_BATCH_SIZE]
            sent_embeds = elmo(elmo_batch)
            all_sent_embeds.append(sent_embeds['elmo_representations'][0])
            all_sent_lengths.append(sent_embeds['mask'].sum(dim=1))

        print(' > ELMo embeddings are calculated for all {} sentences'.format(len(char_ids)))

        self.sent_embeds = torch.cat(all_sent_embeds)
        self.sent_lengths = torch.cat(all_sent_lengths)

    def _embed_bert(self):  # TODO BERT Embeddings
        print('BERT embeddings are not implemented yet. Not using any embeddings')
        # self.sent_embeds = 

    def _cache_data(self, cache_data, filename, embeds):
        if not cache_data:
            return False

        if not os.path.isdir('data'):
            os.mkdir('data')

        if embeds:
            filename += '.' + embeds

        with open(os.path.join('data', filename + '.pickle'), 'wb') as f:
            pickle.dump({
                'sent_tokens': self.sent_tokens,
                'sent_tags': self.sent_tags,
                'sent_embeds': self.sent_embeds,
                'sent_lengths': self.sent_lengths,
                'token_set': self.token_set,
                'char_set': self.char_set,
                'tag_set': self.tag_set,
            }, f)

    def _load_cached_data(self, cache_data, filename, embeds):
        if not cache_data:
            return False

        if embeds:
            filename += '.' + embeds
        filename = os.path.join('data', filename + '.pickle')

        if not os.path.isfile(filename):
            return False

        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.sent_tokens = data['sent_tokens']
            self.sent_tags = data['sent_tags']
            self.sent_embeds = data['sent_embeds']
            self.sent_lengths = data['sent_lengths']
            self.token_set = data['token_set']
            self.char_set = data['char_set']
            self.tag_set = data['tag_set']

        print(':: Loaded cached data from {}'.format(filename))
        return True


class BiLSTMTagger(nn.Module):
    # https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#sphx-glr-beginner-nlp-advanced-tutorial-py
    # https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e

    def __init__(self, w_in_dim, c_in_dim, out_dim,
                 w_emb_dim=128, c_emb_dim=100, hidden_dim=100, c_hidden_dim=100):
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

    def _embed_chars(self, word_iX, char_iX):  # (batch_size, num_words) -> (batch_size, num_words, char_emb_dim)
        # Zero pad char sequences to get equal lengths
        char_x_lengths = torch.tensor([len(x) for x in char_iX])
        char_iX = pad_sequence(char_iX, batch_first=True)

        char_X = self.char_embeds(char_iX)
        char_X = pack_padded_sequence(char_X, char_x_lengths, batch_first=True, enforce_sorted=False)
        _, (char_X, _) = self.char_bilstm(char_X)

        char_X = torch.cat((char_X[0], char_X[1]), dim=1)
        char_X = char_X.view(*word_iX.size(), -1)

        return char_X

    def forward(self, word_iX, char_iX, word_embeds=None):
        word_X = word_embeds
        if word_X is None:
            word_X = self.word_embeds(word_iX)

        char_X = self._embed_chars(word_iX, char_iX)

        assert len(word_X.size()) == 3
        assert len(char_X.size()) == 3
        assert word_X.size()[:2] == char_X.size()[:2]

        X = torch.cat((word_X, char_X), dim=2)
        X, _ = self.bilstm(X)
        X = self.out(X)

        return self.softmax(X)


def train_model(model, train_dataset, epochs=10):
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_total = 0
        epoch_correct = 0

        # TODO Remove sample hack
        # TODO Use DataLoader with batches
        for sent_ix in range(len(train_dataset)):
            model.zero_grad()

            X_words = train_dataset.get_word_tensor(sent_ix).unsqueeze(0)
            X_word_embeds = train_dataset.get_word_embeds(sent_ix)
            if X_word_embeds is not None:
                X_word_embeds = X_word_embeds.unsqueeze(0)
            X_chars = [train_dataset.get_char_tensor(word_ix) for word_ix in X_words.flatten()]
            y = train_dataset.get_tag_tensor(sent_ix).unsqueeze(0)

            # Predict output
            y_pred = model(X_words, X_chars, X_word_embeds)
            loss = loss_function(y_pred.view(-1, len(train_dataset.tag_set)), y.view(-1))

            epoch_total += y.nelement()
            epoch_correct += (y == y_pred.max(dim=2)[1]).sum().item()
            epoch_loss += loss.item()

            # Backpropagate
            loss.backward()
            optimizer.step()

        epoch_losses.append(epoch_loss)
        print(' > Epoch {}/{} acc: {}, loss: {}'.format(epoch + 1, epochs, epoch_correct / epoch_total, epoch_loss))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='ud-treebanks-v2.4/UD_English-ParTUT', help='Path to the conllu files')
    parser.add_argument('--train', default='en_partut-ud-train.conllu',
                        help='training data in conllu format')
    parser.add_argument('--dev', help='validation data in conllu format')
    parser.add_argument('--test', help='test data in conllu format')
    parser.add_argument('--embeds', help='Type of embeddings to use. Choose from [elmo, bert]')
    parser.add_argument('--cache', default=True, help='Cache processed data that is used to train the model')
    parser.add_argument('--epochs', help='number of epochs to train', type=int, default=10)
    args = parser.parse_args()

    if args.embeds == 'elmo':
        print('Using ELMo embeddings\n')
    elif args.embeds == 'bert':
        print('Using BERT embeddings\n')
    else:
        print('Not using pre-trained embeddings\n')

    print('Loading data')
    train_dataset = UniversalDependenciesDataset(args.path, args.train, embeds=args.embeds, cache_data=args.cache)

    print(' > {} sents'.format(len(train_dataset.sent_tokens)))
    print(' > {} unique tokens'.format(len(train_dataset.token_set)))
    print(' > {} unique tags'.format(len(train_dataset.tag_set)))

    print('\nTraining model')
    model = BiLSTMTagger(len(train_dataset.token_set),
                         len(train_dataset.char_set),
                         len(train_dataset.tag_set),
                         w_emb_dim=train_dataset.get_embed_dim(128))
    train_model(model, train_dataset, epochs=args.epochs)


if __name__ == '__main__':
    main()
