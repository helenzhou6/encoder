sweep_configuration = {
    "method": "grid",
    "name": "loss_sweep",
    "metric": {
        "name": "train_loss",
        "goal": "minimize"
    },
    "parameters": {
        "NUM_ENCODER_BLOCKS": {
            "values": [4]
        },
        "NUM_ENCODER_ATTHEADS": {
            "values": [4]
        },
        "NUM_DECODER_BLOCKS": {
            "values": [4]
        },
        "NUM_DECODER_ATTHEADS": {
            "values": [4]
        },
        "EMBEDDING_DIM": {
            "values": [48, 96, 144]
        },
        "PATCH_SIZE": {
            "values": [8, 16, 24]
        }
    }
}