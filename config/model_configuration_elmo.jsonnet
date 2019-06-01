// jsonnet configuration for allennlp model using elmo

local embedding_dim = 6;
local hidden_dim = 6;
local num_epochs = 10;
local patience = 10;
local batch_size = 2;
local learning_rate = 0.1;

{
    "train_data_path": 'ud-treebanks-v2.4/UD_English-ParTUT/en_partut-ud-train.conllu',
    "validation_data_path": 'ud-treebanks-v2.4/UD_English-ParTUT/en_partut-ud-dev.conllu',
    "dataset_reader": {
        "type": "UD",
        "token_indexers": {
            "elmo": {"type": "elmo_characters"}
        }
    },
    "model": {
        "type": "lstm-tagger",
        "word_embeddings": {
            "token_embedders": {               
                "elmo": {
                    "type": "elmo_token_embedder",
                    "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                    "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                    "do_layer_norm": false,
                    "dropout": 0.5
                }
            }
        },
        "encoder": {
          "type": "lstm",
          "input_size": 1024,
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