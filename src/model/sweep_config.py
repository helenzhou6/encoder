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
        "EMBEDDING_DIM": {
            "values": [24,96]
        }
    }
}