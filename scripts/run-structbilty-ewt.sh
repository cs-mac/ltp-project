cd bilstm-aux

# Format the data
grep -v "^#" ../ud-treebanks-v2.4/UD_English-EWT/en_ewt-ud-train.conllu | cut -f 2,4 > data/en_ewt-ud-train.pos
grep -v "^#" ../ud-treebanks-v2.4/UD_English-EWT/en_ewt-ud-dev.conllu | cut -f 2,4 > data/en_ewt-ud-dev.pos
grep -v "^#" ../ud-treebanks-v2.4/UD_English-EWT/en_ewt-ud-test.conllu | cut -f 2,4 > data/en_ewt-ud-test.pos

# Train and test the model
python3 -u src/structbilty.py \
--seed 761 --dynet-autobatch 1 --dynet-mem 4000 \
--train data/en_ewt-ud-train.pos \
--dev data/en_ewt-ud-dev.pos \
--test data/en_ewt-ud-test.pos \
--model en_ewt | tee en_ewt.log
