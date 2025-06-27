import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from model.init_decoder import DigitTransformerDecoder, EncoderDecoderModel
from model.init_encoder import MultiHeadEncoderModel  # assuming encoder is defined here
from utils import get_device, save_artifact
import wandb
from multidigit_dataset import MultiDigitDataset  # custom dataset
from tqdm import tqdm
from sweep_config import sweep_configuration

run_config = {
    "project": "sweeps-on-encoder-decoder", # "digit-transformer" or "sweeps-on-encoder-decoder"
    'run_type': 'sweep',  # 'sweep' or 'train' 
}
default_wandb_config = {
    "model": "EncoderDecoder",
    "NUM_ENCODER_BLOCKS": 4, #6
    "EMBEDDING_DIM": 24, #96
}

# --- Training loop ---
def train():
    wandb.init(project=run_config["project"], config=default_wandb_config)
    config = wandb.config

    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32 #64
    EPOCHS = 20

    ORG_PXL_SIZE = 96  # original image size
    MAX_SEQ_LEN = 6 # <start> max 4 digits (include <pad>) <eod> = total of 6
    OUTPUT_SIZE = 13  # digits 0-9 + <sos>, <eos>, <pad>

    # HYPERPARAMS running sweeps on
    EMBEDDING_DIM = config.EMBEDDING_DIM # dim_model = EMBEDDING_DIM - needs to be divisible by num of heads
    PATCH_SIZE = 4 # 6x6 patches of size 16x16 for 96x96 images
    NUM_CUTS = int(ORG_PXL_SIZE/PATCH_SIZE)
    NUM_PATCHES = int(NUM_CUTS**2)  # 36 patches

    NUM_ENCODER_BLOCKS = config.NUM_ENCODER_BLOCKS
    NUM_ENCODER_ATTHEADS = 4

    NUM_DECODER_BLOCKS = 4
    NUM_DECODER_ATTHEADS = 4

    device = get_device()


    transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor()
    ])

    # --- Dataset ---
    train_data = MultiDigitDataset(data_dir="data/multidigit", split="train")
    val_data = MultiDigitDataset(data_dir="data/multidigit", split="val")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # --- Model ---
    encoder = MultiHeadEncoderModel(num_classes=10, dim_k=EMBEDDING_DIM, num_heads=NUM_ENCODER_ATTHEADS, num_blocks=NUM_ENCODER_BLOCKS)
    decoder = DigitTransformerDecoder(
        vocab_size=OUTPUT_SIZE,
        dim_model=EMBEDDING_DIM,
        num_heads=NUM_DECODER_ATTHEADS,
        num_layers=NUM_DECODER_BLOCKS,
        max_len=MAX_SEQ_LEN
    )
    model = EncoderDecoderModel(encoder, decoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=12)  # ignore <pad> token

    model.train()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    for epoch in range(EPOCHS):
        total_loss, total_acc, total_tokens = 0, 0, 0
        total_digit_correct_train, total_digit_tokens_train = 0, 0
        for batch_imgs, tgt_input, tgt_output in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            batch_imgs = encoder.patch_image_tensor(batch_imgs.to(device))
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            optimizer.zero_grad()
            padding_mask = (tgt_input == 12)
            logits = model(batch_imgs, tgt_input, tgt_key_padding_mask=padding_mask)
            loss = loss_fn(logits.view(-1, OUTPUT_SIZE), tgt_output.view(-1))

            loss.backward()

            #for name, param in model.named_parameters():
            #    if param.grad is not None:
            #        wandb.log({f"grad_norm/{name}": param.grad.norm().item()})

            optimizer.step()

            total_loss += loss.item() * batch_imgs.size(0)
            preds = logits.argmax(dim=-1)
            # DEBUG: Print distribution of predicted digit tokens (0–9) only
            preds_flat = preds.view(-1)
            pred_digit_mask = (preds_flat >= 0) & (preds_flat <= 9)
            pred_digits = preds_flat[pred_digit_mask]
            print("Predicted digit distribution:", torch.bincount(pred_digits, minlength=10).cpu().numpy())

            tgt_flat = tgt_output.view(-1)
            tgt_digit_mask = (tgt_flat >= 0) & (tgt_flat <= 9)
            tgt_digits = tgt_flat[tgt_digit_mask]
            print("Target digit distribution:", torch.bincount(tgt_digits, minlength=10).cpu().numpy())

            #Token Accuracy (ignoring <pad> token)
            correct = preds.view(-1) == tgt_output.view(-1)
            total_acc += correct.sum().item()
            total_tokens += (tgt_output != 12).sum().item()
            # Digit Accuracy (0-9)
            digit_mask = (tgt_output >= 0) & (tgt_output <= 9)
            digit_correct = ((preds == tgt_output) & digit_mask).sum().item()
            digit_total = digit_mask.sum().item()
            total_digit_correct_train += digit_correct
            total_digit_tokens_train += digit_total

        # Validation
        model.eval()
        val_loss, val_acc, val_tokens = 0, 0, 0
        total_digit_correct_val, total_digit_tokens_val = 0, 0
        with torch.no_grad():
            for batch_imgs, tgt_input, tgt_output in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                batch_imgs = encoder.patch_image_tensor(batch_imgs.to(device))
                tgt_input = tgt_input.to(device)
                tgt_output = tgt_output.to(device)

                padding_mask = (tgt_input == 12)
                logits = model(batch_imgs, tgt_input, tgt_key_padding_mask=padding_mask)
                loss = loss_fn(logits.view(-1, OUTPUT_SIZE), tgt_output.view(-1))
                
                val_loss += loss.item() * batch_imgs.size(0)
                preds = logits.argmax(dim=-1)
                # DEBUG: Print distribution of predicted digit tokens (0–9) only
                preds_flat = preds.view(-1)
                pred_digit_mask = (preds_flat >= 0) & (preds_flat <= 9)
                pred_digits = preds_flat[pred_digit_mask]
                print("Predicted digit distribution:", torch.bincount(pred_digits, minlength=10).cpu().numpy())

                tgt_flat = tgt_output.view(-1)
                tgt_digit_mask = (tgt_flat >= 0) & (tgt_flat <= 9)
                tgt_digits = tgt_flat[tgt_digit_mask]
                print("Target digit distribution:", torch.bincount(tgt_digits, minlength=10).cpu().numpy())

                correct = preds.view(-1) == tgt_output.view(-1)
                val_acc += correct.sum().item()
                val_tokens += (tgt_output != 12).sum().item()

                digit_mask = (tgt_output >= 0) & (tgt_output <= 9)
                digit_correct = ((preds == tgt_output) & digit_mask).sum().item()
                digit_total = digit_mask.sum().item()
                total_digit_correct_val += digit_correct
                total_digit_tokens_val += digit_total


            try:
                wandb.log({
                    "train_loss": total_loss / len(train_data),
                    "train_acc": total_acc / total_tokens,
                    "val_loss": val_loss / len(val_data),
                    "val_acc": val_acc / val_tokens,
                    "digit_train_acc": total_digit_correct_train / total_digit_tokens_train,
                    "digit_val_acc": total_digit_correct_val / total_digit_tokens_val,
                })
            except Exception as e:
                print(f"[wandb log failed] {e}")
            # Visualize a few predictions
            if epoch % 5 == 0:
                print("\nSample predictions:")
                for i in range(min(3, preds.size(0))):
                    print("Predicted:", preds[i].tolist())
                    print("Target:   ", tgt_output[i].tolist())

        print(f"Epoch {epoch+1} | Train Loss: {total_loss:.3f} | Train Acc: {total_acc/total_tokens*100:.2f}%"
              f" | Val Loss: {val_loss:.3f} | Val Acc: {val_acc/val_tokens*100:.2f}%")
        scheduler.step()
        wandb.log({"learning_rate": scheduler.get_last_lr()[0]})
    # Save locally
    model_path = "sweep_transformer.pt"
    torch.save(model.state_dict(), model_path)

    # Log to wandb as an artifact
    artifact = wandb.Artifact("sweep_transformer", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    try:
        wandb.finish()
    except Exception as e:
        print(f"[wandb finish failed] {e}")


if __name__ == "__main__":
    if run_config["run_type"] == "sweep":
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=run_config["project"])
        wandb.agent(
            sweep_id=sweep_id,
            function=train,
            project=run_config['project'],
            count=4,
        )
    elif run_config["run_type"] == "train":
        train()
