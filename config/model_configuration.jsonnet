// jsonnet configuration for allennlp model using simple word embedder

local embedding_dim = 6;
local hidden_dim = 6;
local num_epochs = 200;
local patience = 10;
local batch_size = 2;
local learning_rate = 0.1;

{
    "train_data_path": 'ud-treebanks-v2.4/UD_English-ParTUT/en_partut-ud-train.conllu',
    "validation_data_path": 'ud-treebanks-v2.4/UD_English-ParTUT/en_partut-ud-dev.conllu',
    "dataset_reader": {
        "type": "UD"
    },
    "model": {
        "type": "lstm-tagger",
        "word_embeddings": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": embedding_dim
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": hidden_dim
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": [["sentence", "num_tokens"]]
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "sgd",
            "lr": learning_rate
        },
        "patience": patience
    }
}