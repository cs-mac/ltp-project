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
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from allennlp.modules.elmo import Elmo, batch_to_ids
from pytorch_pretrained_bert import BertTokenizer, BertModel

torch.manual_seed(673)


ELMO_OPTIONS_FILE = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json'
ELMO_WEIGHT_FILE = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

ELMO_BATCH_SIZE = 64
BERT_BATCH_SIZE = 32


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


def retokenize_bert(sents, og_tags):
    '''
    Transform data using BERT Wordpieces, and [CLS], [SEP] tags
    '''
    new_sent_ids, new_sent_tokens, new_tags = [], [], []
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_basic_tokenize=False, do_lower_case=False)
    for sentence, tags in zip(sents, og_tags):
        tok_sentence = []
        corrected_tags = []
        for word, tag in zip(sentence, tags):
            tokenized_text = tokenizer.tokenize(word)
            if len(tokenized_text) > 1:
                tok_sentence.append(tokenized_text[0])
                corrected_tags.append(tag)
                for i in range(1, len(tokenized_text)):
                    tok_sentence.append(tokenized_text[i])
                    corrected_tags.append('<PAD>')
            else:
                tok_sentence.append(''.join(tokenized_text))
                corrected_tags.append(tag)

        indexed_tokens = tokenizer.convert_tokens_to_ids(['[CLS]']+tok_sentence+['[SEP]'])

        new_sent_tokens.append(tok_sentence)
        new_tags.append(corrected_tags)
        new_sent_ids.append(indexed_tokens)
    return new_sent_tokens, new_tags, new_sent_ids


class UniversalDependenciesDataset(Dataset):
    def __init__(self, root_path, filename, idx_map=None, embeds=None, cache_data=False):
        self.filename = filename
        self.filepath = os.path.join(root_path, filename)
        self.embeds = embeds
        self.cache = cache_data

        self.sent_tokens = []
        self.sent_tags = []

        self.word_tensor = None
        self.char_tensor = None
        self.tag_tensor = None
        self.sent_lengths = None
        self.word_lengths = None

        self.token_set = ['<PAD>']
        self.char_set = ['<PAD>'] + [chr(i) for i in range(128)]
        self.tag_set = ['<PAD>']

        loaded_cache = self._load_cached_data()
        if not loaded_cache:
            self._read_conllu()

        self.token_to_ix = idx_map['token'] if idx_map else self._create_value_map(self.token_set)
        self.char_to_ix = idx_map['char'] if idx_map else self._create_value_map(self.char_set)
        self.tag_to_ix = idx_map['tag'] if idx_map else self._create_value_map(self.tag_set)

        if not loaded_cache:
            self._prepare_word_tensor()
            self._prepare_char_tensor()
            self._prepare_tag_tensor()
            self._cache_data()

    def __len__(self):
        return len(self.sent_tokens)

    def __getitem__(self, idx):
        return self.word_tensor[idx], self.char_tensor[idx], \
            self.sent_lengths[idx], self.word_lengths[idx], self.tag_tensor[idx]

    def get_embed_dim(self, default=None):
        if len(self.word_tensor.size()) == 3:
            return self.word_tensor.size()[2]
        return default

    def get_idx_map(self):
        return {
            'token': self.token_to_ix,
            'char': self.char_to_ix,
            'tag': self.tag_to_ix,
        }

    def _create_value_map(self, vals):
        return {val: ix for ix, val in enumerate(vals)}

    def _read_conllu(self):
        for tokens, tags in conllu_reader(self.filepath):
            self.sent_tokens.append(tokens)
            self.sent_tags.append(tags)

            for token in tokens:
                if token not in self.token_set:
                    self.token_set.append(token)
            for tag in tags:
                if tag not in self.tag_set:
                    self.tag_set.append(tag)

    def _prepare_word_tensor(self):
        if not self.embeds:
            return self._word_embed_idx()
        if self.embeds == 'elmo':
            return self._word_embed_elmo()
        if self.embeds == 'bert':
            return self._word_embed_bert()
        print('Unknown embedding type: {}'.format(self.embeds))

    def _word_embed_idx(self):
        sent_tensors = [torch.tensor([self.token_to_ix[t] if t in self.token_to_ix else 0 for t in sent])
                        for sent in self.sent_tokens]
        self.word_tensor = pad_sequence(sent_tensors, batch_first=True)
        self.sent_lengths = torch.tensor([len(sent) for sent in self.sent_tokens])

    def _word_embed_elmo(self):
        print(':: Initializing ELMo')
        elmo = Elmo(ELMO_OPTIONS_FILE, ELMO_WEIGHT_FILE, 1, dropout=0)

        print(':: Encoding tokens')
        char_ids = batch_to_ids(self.sent_tokens)

        print(':: Calculating ELMo embeddings')
        sent_embeds, sent_lengths = [], []
        for i in range(0, len(char_ids), ELMO_BATCH_SIZE):
            print(' > {}/{} [{}%]'.format(i, len(char_ids), int(i / len(char_ids) * 100)), end='\r')

            elmo_batch = char_ids[i:i+ELMO_BATCH_SIZE]
            sent_batch_embeds = elmo(elmo_batch)
            sent_embeds.append(sent_batch_embeds['elmo_representations'][0])
            sent_lengths.append(sent_batch_embeds['mask'].sum(dim=1))

        print(' > ELMo embeddings are calculated for all {} sentences'.format(len(char_ids)))

        self.word_tensor = torch.cat(sent_embeds)
        self.sent_lengths = torch.cat(sent_lengths)

    def _word_embed_bert(self):
        print(':: Retokenizing and encoding tokens')
        self.sent_tokens, self.sent_tags, token_ids = retokenize_bert(self.sent_tokens, self.sent_tags)
        self.sent_lengths = torch.tensor([len(sent) for sent in self.sent_tokens])
        self.token_set = list(set([t for sent in self.sent_tokens for t in sent]))
        self.token_to_ix = self._create_value_map(self.token_set)

        token_tensor = pad_sequence([torch.tensor(sent) for sent in token_ids], batch_first=True)

        print(':: Initializing BERT')
        bert = BertModel.from_pretrained('bert-base-cased')

        print(':: Calculating BERT embeddings')
        all_sent_embeds = []
        for i in range(0, len(token_tensor), BERT_BATCH_SIZE):
            print(' > {}/{} [{}%]'.format(i, len(token_tensor), int(i / len(token_tensor) * 100)), end='\r')

            with torch.no_grad():
                bert_batch = token_tensor[i:i+BERT_BATCH_SIZE]
                encoded_layers, _ = bert(bert_batch)
                encoded_layer = encoded_layers[-1][:, 1:-1]

            all_sent_embeds.append(encoded_layer)

        self.word_tensor = torch.cat(all_sent_embeds)

    def _prepare_char_tensor(self):
        def normalize(t):
            return unicodedata.normalize('NFKD', t)

        print(':: Calculating character embeddings')

        token_set = set([normalize(t) for t in self.token_set])
        tensor_map = {t: torch.tensor([self.char_to_ix[c] if c in self.char_to_ix else 0
                                       for c in t]) for t in token_set}

        char_tensor = pad_sequence([tensor_map[normalize(t)]
                                    for sent in self.sent_tokens for t in sent], batch_first=True)
        word_lengths = torch.tensor([len(normalize(t)) for sent in self.sent_tokens for t in sent])

        # Divide char embeddings back into sentences
        sent_tensors, sent_word_lengths, prev_i = [], [], 0
        for i in self.sent_lengths:
            sent_tensors.append(char_tensor[prev_i:prev_i + i])
            sent_word_lengths.append(word_lengths[prev_i:prev_i + i])
            prev_i += i
        assert prev_i == self.sent_lengths.sum().item()

        self.char_tensor = pad_sequence(sent_tensors, batch_first=True)
        self.word_lengths = pad_sequence(sent_word_lengths, batch_first=True)

    def _prepare_tag_tensor(self):
        sent_tensors = [torch.tensor([self.tag_to_ix[t] if t in self.tag_to_ix else 0 for t in sent])
                        for sent in self.sent_tags]
        self.tag_tensor = pad_sequence(sent_tensors, batch_first=True)

    def _get_cache_path(self):
        if not os.path.isdir('data'):
            os.mkdir('data')

        filepath = os.path.join('data', self.filename)
        if self.embeds:
            filepath += '.' + self.embeds
        return filepath + '.pickle'

    def _cache_data(self):
        if not self.cache:
            return False

        with open(self._get_cache_path(), 'wb') as f:
            pickle.dump({
                'token_set': self.token_set,
                'char_set': self.char_set,
                'tag_set': self.tag_set,

                'sent_tokens': self.sent_tokens,
                'sent_tags': self.sent_tags,

                'word_tensor': self.word_tensor,
                'char_tensor': self.char_tensor,
                'tag_tensor': self.tag_tensor,
                'sent_lengths': self.sent_lengths,
                'word_lengths': self.word_lengths,
            }, f)

    def _load_cached_data(self):
        if not self.cache:
            return False

        cache_path = self._get_cache_path()
        if not os.path.isfile(cache_path):
            return False

        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            self.token_set = data['token_set']
            self.char_set = data['char_set']
            self.tag_set = data['tag_set']

            self.sent_tokens = data['sent_tokens']
            self.sent_tags = data['sent_tags']

            self.word_tensor = data['word_tensor']
            self.char_tensor = data['char_tensor']
            self.tag_tensor = data['tag_tensor']
            self.sent_lengths = data['sent_lengths']
            self.word_lengths = data['word_lengths']

        print(':: Loaded cached data from {}'.format(cache_path))
        return True


class BiLSTMTagger(nn.Module):
    # https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#sphx-glr-beginner-nlp-advanced-tutorial-py
    # https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e

    def __init__(self, w_in_dim, c_in_dim, out_dim, w_emb_dim, c_emb_dim, w_dropout, c_dropout, noise,
                 hidden_dim=100, c_hidden_dim=100):  # TODO Add use_word_features, use_char_features booleans
        super(BiLSTMTagger, self).__init__()

        self.noise = noise

        # Word embedding
        self.word_embeds = nn.Embedding(w_in_dim, w_emb_dim)
        self.word_dropout = nn.Dropout(w_dropout)

        # Char embedding
        self.char_embeds = nn.Embedding(c_in_dim, c_emb_dim)
        self.char_dropout = nn.Dropout(c_dropout)
        self.char_bilstm = nn.LSTM(c_emb_dim, c_hidden_dim, num_layers=1,
                                   batch_first=True, bidirectional=True)

        # Network
        self.bilstm = nn.LSTM(w_emb_dim + 2 * c_emb_dim, hidden_dim, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_dim * 2, out_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def _embed_chars(self, X_chars, word_lengths, is_training):
        X_chars = self.char_embeds(X_chars)

        if is_training:
            X_chars = self.char_dropout(X_chars)

        # Flatten batch and sentence dimensions
        X_chars = X_chars.view(-1, *X_chars.size()[2:])
        flat_word_lengths = word_lengths.flatten()
        flat_mask = flat_word_lengths.gt(0)

        # Mask all non-word rows
        masked_X_chars = X_chars[flat_mask]
        masked_word_lengths = flat_word_lengths[flat_mask]
        masked_X_chars = pack_padded_sequence(masked_X_chars, masked_word_lengths,
                                              batch_first=True, enforce_sorted=False)
        _, (masked_X_chars, _) = self.char_bilstm(masked_X_chars)
        masked_X_chars = torch.cat((masked_X_chars[0], masked_X_chars[1]), dim=1)

        # Unmask and unflatten
        X_chars = torch.zeros(flat_mask.size()[0], masked_X_chars.size()[-1])
        X_chars[flat_mask] = masked_X_chars
        X_chars = X_chars.view(*word_lengths.size(), X_chars.size()[-1])
        return X_chars

    def forward(self, X_words, X_chars, sent_lengths, word_lengths, is_training=True):
        assert X_words.size()[:2] == X_chars.size()[:2]
        assert X_words.size()[0] == sent_lengths.size()[0]
        assert X_chars.size()[:2] == word_lengths.size()[:2]

        # Embed words if they are not embedded yet
        if len(X_words.size()) == 2:
            X_words = self.word_embeds(X_words)
        else:
            X_words.requires_grad_(False)

        if is_training:
            X_words = self.word_dropout(X_words)

        # Embed characters
        X_chars = self._embed_chars(X_chars, word_lengths, is_training)

        # Concatenate Word & Char embeddings
        assert len(X_words.size()) == 3
        assert len(X_chars.size()) == 3
        X = torch.cat((X_words, X_chars), dim=2)

        # Add noise during training
        if is_training:
            X_noise = torch.zeros(*X.size()).normal_(std=self.noise).detach()
            X = X + X * X_noise

        # Predict POS tags
        packed_X = pack_padded_sequence(X, sent_lengths, batch_first=True, enforce_sorted=False)
        packed_X, _ = self.bilstm(packed_X)
        X, _ = pad_packed_sequence(packed_X, batch_first=True)
        X = self.out(X)

        return self.softmax(X)


def run_batch(model, batch, is_training=True):
    X_words, X_chars, sent_lengths, word_lengths, y_tags = batch
    batch_mask = y_tags.sum(dim=0).gt(0)

    X_words, X_chars, word_lengths, y_tags = \
        X_words[:, batch_mask], X_chars[:, batch_mask], word_lengths[:, batch_mask], y_tags[:, batch_mask]

    y_mask = y_tags.gt(0)

    # Predict output
    y_pred = model(X_words, X_chars, sent_lengths, word_lengths, is_training)
    return y_pred[y_mask], y_tags[y_mask]


def train_model(model, train_dataset, valid_dataset=None, batch_size=16, epochs=10):
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    if valid_dataset:
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    print(':: Starting training')
    verbose_interval = max(1, len(train_dataloader) // 50)

    prev_valid_correct = 0

    for epoch in range(epochs):
        epoch_loss, epoch_total, epoch_correct = 0, 0, 0

        # Training
        for batch_i, batch in enumerate(train_dataloader):
            model.zero_grad()

            y_pred, y_true = run_batch(model, batch)

            loss = loss_function(y_pred, y_true)

            epoch_total += y_true.nelement()
            epoch_correct += (y_true == y_pred.max(dim=1)[1]).sum().item()
            epoch_loss += loss.item()

            # Backpropagate
            loss.backward()
            optimizer.step()

            if batch_i % verbose_interval == 0:
                out_line = ' > Epoch {:2d}/{}: {}/{} batches [{}%]'
                print(out_line.format(epoch + 1, epochs, batch_i, len(train_dataloader),
                                      int(batch_i / len(train_dataloader) * 100)), end='\r')

        # Validation
        valid_total, valid_correct = 0, 0
        if valid_dataset:
            with torch.set_grad_enabled(False):
                for batch_i, batch in enumerate(valid_dataloader):
                    y_pred, y_true = run_batch(model, batch, is_training=False)

                    valid_total += y_true.nelement()
                    valid_correct += (y_true == y_pred.max(dim=1)[1]).sum().item()

        out_line = ' > Epoch {:2d}/{} train loss: {:.4f}, train acc: {:.4f}, valid acc: {:.4f} [{}{:.4f}]'
        print(out_line.format(epoch + 1, epochs, epoch_loss / len(train_dataloader),
                              epoch_correct / epoch_total, valid_correct / valid_total,
                              '+' if valid_correct > prev_valid_correct else '',
                              (valid_correct - prev_valid_correct) / valid_total))
        prev_valid_correct = valid_correct


def print_data(name, data):
    print('\n### {} ###'.format(name))
    for key, val in data.items():
        print(' > {} {}'.format(key, val))
    print('')


def summarize_dataset(datatype, dataset):
    print_data('{} Dataset'.format(datatype), {
        'filepath       ': dataset.filepath,
        '# sents        ': len(dataset.sent_tokens),
        '# unique tokens': len(dataset.token_set),
        '# unique tags  ': len(dataset.tag_set),
    })


def summarize_training_args(args, w_emb_size):
    print_data('Model training options', {
        'Word embeddings  ': args.embeds.upper() if args.embeds else 'Not pre-trained',
        'Word emb size    ': w_emb_size,
        'Char emb size    ': args.char_embed_size,
        'word dropout rate': args.word_dropout,
        'char dropout rate': args.char_dropout,
        'gaussian noise   ': args.noise,
        '# epochs         ': args.epochs,
        'batch size       ': args.batch,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='ud-treebanks-v2.4/UD_English-ParTUT', help='Path to the conllu files')
    parser.add_argument('--train', default='en_partut-ud-train.conllu',
                        help='training data in conllu format')
    parser.add_argument('--dev', default='en_partut-ud-dev.conllu', help='validation data in conllu format')
    parser.add_argument('--test', default='en_partut-ud-test.conllu', help='test data in conllu format')
    parser.add_argument('--cache', default=True, help='Cache processed data that is used to train the model')
    parser.add_argument('--embeds', help='Type of pre-trained embeddings to use. Choose from [elmo, bert]')
    parser.add_argument(
        '--word-embed-size', help='word embedding size, only used when not using pre-trained embeddings',
        type=int, default=256)
    parser.add_argument('--char-embed-size', help='char embedding size', type=int, default=100)
    parser.add_argument('--epochs', help='number of epochs to train', type=int, default=20)
    parser.add_argument('--batch', help='batch size', type=int, default=16)
    parser.add_argument('--word-dropout', help='word dropout rate', type=float, default=0.25)
    parser.add_argument('--char-dropout', help='char dropout rate', type=float, default=0.0)
    parser.add_argument('--noise', help='sigma of Gaussian noise', type=float, default=0.2)
    args = parser.parse_args()

    print('# Loading training data')
    train_dataset, valid_dataset, test_dataset = None, None, None

    train_dataset = UniversalDependenciesDataset(args.path, args.train, embeds=args.embeds, cache_data=args.cache)
    summarize_dataset('Train', train_dataset)

    if args.dev:
        print('# Loading validation data')
        valid_dataset = UniversalDependenciesDataset(args.path, args.dev, idx_map=train_dataset.get_idx_map(),
                                                     embeds=args.embeds, cache_data=args.cache)
        summarize_dataset('Validation', valid_dataset)

    print('# Training model')
    w_emb_size = train_dataset.get_embed_dim(args.word_embed_size)
    summarize_training_args(args, w_emb_size)

    model = BiLSTMTagger(len(train_dataset.token_set),
                         len(train_dataset.char_set),
                         len(train_dataset.tag_set),
                         w_emb_dim=w_emb_size,
                         c_emb_dim=args.char_embed_size,
                         w_dropout=args.word_dropout,
                         c_dropout=args.char_dropout,
                         noise=args.noise,
                         )

    train_model(model, train_dataset, valid_dataset, batch_size=args.batch, epochs=args.epochs)

    if args.test:
        print('# Loading testing data')
        test_dataset = UniversalDependenciesDataset(args.path, args.test, idx_map=train_dataset.get_idx_map(),
                                                    embeds=args.embeds, cache_data=args.cache)
        summarize_dataset('Test', test_dataset)

        print('# Testing model')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch)
        test_total, test_correct = 0, 0
        with torch.set_grad_enabled(False):
            for batch_i, batch in enumerate(test_dataloader):
                y_pred, y_true = run_batch(model, batch, is_training=False)

                # train_dataset.tag_set[5]

                test_total += y_true.nelement()
                test_correct += (y_true == y_pred.max(dim=1)[1]).sum().item()
            print(' > Test accuracy: {:.4f}'.format(test_correct / test_total))


if __name__ == '__main__':
    main()
