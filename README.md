# Language Technology Project - POS tagger

## Requirements

 - Python 3.6+
 - The Python packages that are listed in `requirements.txt`
 - The data files:
   - Run `./prepare.sh` once to download the required files

## Progress

 - [x] Wietse: Write simple baseline PyTorch model
 - [ ] Corbèn: Read data from `ud-treebanks-v2.4/UD_English-ParTUT/en_partut_ud-train.conllu` and use the data in the baseline model.
 - [ ] Chi Sam: Add ELMo/BERT to baseline model instead of simple embedding layer.
 - [ ] Wietse: Research and write model architecture based on [structbilty](https://github.com/bplank/bilstm-aux).
 - [ ] Model output evaluation
