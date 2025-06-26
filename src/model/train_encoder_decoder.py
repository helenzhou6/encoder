import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from model.init_decoder import DigitTransformerDecoder, EncoderDecoderModel
from model.init_encoder import MultiHeadEncoderModel  # assuming encoder is defined here
from utils import get_device
import wandb
from multidigit_dataset import MultiDigitDataset  # custom dataset
from tqdm import tqdm

# --- Config ---
BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 0.0005

ORG_PXL_SIZE = 96  # original image size
PATCH_SIZE = 16 # 6x6 patches of size 16x16 for 96x96 images
NUM_CUTS = int(ORG_PXL_SIZE/PATCH_SIZE)
NUM_PATCHES = int(NUM_CUTS**2)  # 36 patches
EMBEDDING_DIM = 96 # dim_model = EMBEDDING_DIM

MAX_SEQ_LEN = 6
OUTPUT_SIZE = 13  # digits 0-9 + <sos>, <eos>, <pad>

NUM_ENCODER_BLOCKS = 4
NUM_ENCODER_ATTHEADS = 4

NUM_DECODER_BLOCKS = 4
NUM_DECODER_ATTHEADS = 4

device = get_device()
wandb.init(project="digit-transformer")

# --- Positional patch encoder ---
linear_proj = nn.Linear(PATCH_SIZE * PATCH_SIZE, EMBEDDING_DIM)
row_embed = nn.Embedding(NUM_CUTS, EMBEDDING_DIM // 2)
col_embed = nn.Embedding(NUM_CUTS, EMBEDDING_DIM // 2)


transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor()
])

def patch_image_tensor(img_tensor):
    patches = img_tensor.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)  # (B, channel, division, division, patch_width, patch_height)
    patches = patches.contiguous().view(img_tensor.size(0), NUM_PATCHES, -1)  # (B, NUM_PATCHES, patch_size * patch_size)
    
    patch_embeddings = linear_proj(patches)  # (B, NUM_PATCHES, EMBEDDING_DIM)

    # Add positional encoding
    positions = torch.arange(NUM_PATCHES, device=img_tensor.device)
    rows = positions // NUM_CUTS
    cols = positions % NUM_CUTS
    pos_embed = torch.cat([row_embed(rows), col_embed(cols)], dim=1)  # (NUM_PATCHES, EMBEDDING_DIM)
    patch_embeddings = patch_embeddings + pos_embed.unsqueeze(0)  # (1, NUM_PATCHES, EMBEDDING_DIM)

    return patch_embeddings


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

# --- Training loop ---
def train():
    model.train()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    for epoch in range(EPOCHS):
        total_loss, total_acc, total_tokens = 0, 0, 0
        total_digit_correct_train, total_digit_tokens_train = 0, 0
        for batch_imgs, tgt_input, tgt_output in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            batch_imgs = patch_image_tensor(batch_imgs.to(device))
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
                batch_imgs = patch_image_tensor(batch_imgs.to(device))
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


            wandb.log({
                "train_loss": total_loss / len(train_data),
                "train_acc": total_acc / total_tokens,
                "val_loss": val_loss / len(val_data),
                "val_acc": val_acc / val_tokens,
                "digit_train_acc": total_digit_correct_train / total_digit_tokens_train,
                "digit_val_acc": total_digit_correct_val / total_digit_tokens_val,
            })
            # Visualize a few predictions
            if epoch % 1 == 0:
                print("\nSample predictions:")
                for i in range(min(3, preds.size(0))):
                    print("Predicted:", preds[i].tolist())
                    print("Target:   ", tgt_output[i].tolist())

        print(f"Epoch {epoch+1} | Train Loss: {total_loss:.3f} | Train Acc: {total_acc/total_tokens*100:.2f}%"
              f" | Val Loss: {val_loss:.3f} | Val Acc: {val_acc/val_tokens*100:.2f}%")
        scheduler.step()
        wandb.log({"learning_rate": scheduler.get_last_lr()[0]})
    torch.save(model.state_dict(), "digit_transformer.pt")

if __name__ == "__main__":
    train()
