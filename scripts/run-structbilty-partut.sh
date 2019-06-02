cd bilstm-aux

# Format the data
grep -v "^#" ../ud-treebanks-v2.4/UD_English-ParTUT/en_partut-ud-train.conllu | cut -f 2,4 > data/en_partut-ud-train.pos
grep -v "^#" ../ud-treebanks-v2.4/UD_English-ParTUT/en_partut-ud-dev.conllu | cut -f 2,4 > data/en_partut-ud-dev.pos
grep -v "^#" ../ud-treebanks-v2.4/UD_English-ParTUT/en_partut-ud-test.conllu | cut -f 2,4 > data/en_partut-ud-test.pos

# Train and test the model
python3 -u src/structbilty.py \
--seed 761 --dynet-autobatch 1 --dynet-mem 4000 \
--train data/en_partut-ud-train.pos \
--dev data/en_partut-ud-dev.pos \
--test data/en_partut-ud-test.pos \
--model en_partut | tee en_partut.log
