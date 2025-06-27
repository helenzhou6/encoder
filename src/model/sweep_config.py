sweep_configuration = {
    "method": "grid",
    "name": "loss_sweep",
    "metric": {
        "name": "train_acc",
        "goal": "maximize"
    },
    "parameters": {
        "NUM_ENCODER_BLOCKS": {
            "values": [4,6]
        },
        "NUM_ENCODER_ATTHEADS": {
            "values": [4,6]
        },
        "NUM_DECODER_BLOCKS": {
            "values": [4,6]
        },
        "NUM_DECODER_ATTHEADS": {
            "values": [4,6]
        },
        "EMBEDDING_DIM": {
            "values": [24, 48, 96]
        },
        "PATCH_SIZE": {
            "values": [4, 8, 16]
        }
    }
}