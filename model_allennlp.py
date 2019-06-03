# #!/usr/bin/env python3

# Derived from: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# Reader derived from: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/universal_dependencies.py
# Predictor derived from: http://mlexplained.com/2019/01/30/an-in-depth-tutorial-to-allennlp-from-basics-to-elmo-and-bert/

# TODO:
# REORDER / REMOVE REDUNDANT IMPORTS
# CREATE EVALUATION (F-metric, confusion matrix)
# ADD BERT CONFIGURATION JSON
# WHAT MODEL SIZE? / OTHER PARAMETERS
# CREATE NICER PREDICTOR SOLUTION
# DOCUMENTATION
# ADD TO THE TODO

import sys
from typing import Iterator, List, Dict, Tuple, Iterable
import shutil
import tempfile

from conllu.parser import parse_line, DEFAULT_FIELDS
from overrides import overrides

import torch
import numpy as np

from allennlp.commands.train import train_model
from allennlp.common.params import Params
from allennlp.common.file_utils import cached_path
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import DataIterator
from tqdm import tqdm
from scipy.special import expit # the sigmoid function
from allennlp.data.iterators import BasicIterator
from allennlp.nn import util as nn_util
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer

torch.manual_seed(1)

def lazy_parse(text: str, fields: Tuple[str, ...]=DEFAULT_FIELDS):
    for sentence in text.split("\n\n"):
        if sentence:
            yield [parse_line(line, fields)
                   for line in sentence.split("\n")
                   if line and not line.strip().startswith("#")]

@DatasetReader.register('UD')
class UniversalDependenciesDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)

        with open(file_path, 'r') as conllu_file:

            for annotation in  lazy_parse(conllu_file.read()):
                annotation = [x for x in annotation if x["id"] is not None]

                words = [x["form"] for x in annotation]
                pos_tags = [x["upostag"] for x in annotation]
                # print(f'Sentence =\n{words}\n{pos_tags}\n')
                # print(f'Sentence =\n{len(words)}\n{len(pos_tags)}\n')
                yield self.text_to_instance(words, pos_tags)

    @overrides
    def text_to_instance(self,
                         words: List[str],
                         upos_tags: List[str] = None) -> Instance:

        fields: Dict[str, Field] = {}

        sentence_field = TextField([Token(w) for w in words], self._token_indexers)
        fields = {"sentence": sentence_field}

        if upos_tags:
            pos_tag_field = SequenceLabelField(labels=upos_tags, sequence_field=sentence_field)
            fields["labels"] = pos_tag_field

        return Instance(fields)


@Model.register('lstm-tagger')
class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: BasicTextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))

        self.accuracy = CategoricalAccuracy()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}

        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


def tonp(tsr): return tsr.detach().cpu().numpy()

class Predictor:
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device

    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        return expit(tonp(out_dict["tag_logits"]))

    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                preds.append(self._extract_data(batch))
        return np.concatenate(preds, axis=0)


if __name__ == "__main__":
    token_ind = False
    # Choose which configuration file to use
    try:
        if sys.argv[1] == "base":
            params = Params.from_file('config/model_configuration.jsonnet')
        elif sys.argv[1] == "elmo":
            params = Params.from_file('config/model_configuration_elmo.jsonnet')
            token_ind = {"elmo": ELMoTokenCharactersIndexer()}
        elif sys.argv[1] == "bert":
            params = Params.from_file('config/model_configuration_bert.jsonnet')
    except IndexError:
        print("No argument given: using base model! \nPossible arguments are 'base', 'elmo' or 'bert'")
        params = Params.from_file('config/model_configuration.jsonnet')

    # Remove a saved model, when the model is going to run again
    try:
        shutil.rmtree("created_model")
    except FileNotFoundError:
        pass

    print("\n###################### STARTING ############################\n")

    model = train_model(params, "created_model")

    print("\n###################### PREDICTING ############################\n")

    seq_iterator = BasicIterator(batch_size=64)
    seq_iterator.index_with(model.vocab)

    predictor = Predictor(model, seq_iterator)
    if token_ind:
        reader = UniversalDependenciesDatasetReader(token_ind)
    else:
        reader = UniversalDependenciesDatasetReader()

    # Maybe try changing predictor in such a way the reader is not a necessary step here, but just the file will do
    tag_logits = predictor.predict(reader.read("ud-treebanks-v2.4/UD_English-ParTUT/en_partut-ud-dev.conllu"))
    tag_ids = np.argmax(tag_logits, axis=-1)

    for instance in tag_ids:
        print([model.vocab.get_token_from_index(i, 'labels') for i in instance])
