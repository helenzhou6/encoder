import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torchmetrics import Accuracy
from model.init_model_decoder import DigitTransformerDecoder, EncoderDecoderModel
from model.init_model_mulhd import MultiHeadEncoderModel  # assuming encoder is defined here
from utils import get_device
import wandb
from data.multidigit_dataset import MultiDigitDataset  # custom dataset
from tqdm import tqdm

# --- Config ---
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.0005

PATCH_SIZE = 24 # 4x4 patches of size 24x24 for 96x96 images
EMBEDDING_DIM = 96

SEQ_LEN = 6
VOCAB_SIZE = 13  # digits 0-9 + <sos>, <eos>, <pad>

device = get_device()
wandb.init(project="digit-transformer")

# --- Positional patch encoder ---
linear_proj = nn.Linear(PATCH_SIZE * PATCH_SIZE, EMBEDDING_DIM)
row_embed = nn.Embedding(4, EMBEDDING_DIM // 2)
col_embed = nn.Embedding(4, EMBEDDING_DIM // 2)

def patch_image(image):  # [1, 28, 28]
    patches = image.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
    patches = patches.contiguous().view(-1, PATCH_SIZE * PATCH_SIZE)
    patch_embeddings = linear_proj(patches)
    positions = torch.arange(16, device=patch_embeddings.device)
    rows = positions // 4
    cols = positions % 4
    pos_embed = torch.cat([row_embed(rows), col_embed(cols)], dim=1)
    patch_embeddings = patch_embeddings + pos_embed
    return patch_embeddings

transform = transforms.Compose([
    transforms.ToTensor()
])

def patch_image_tensor(img_tensor):
    # Input: (B, 1, 28, 28)
    patches = img_tensor.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)  # (B, 1, 4, 4, 7, 7)
    patches = patches.contiguous().view(img_tensor.size(0), 16, -1)  # (B, 16, 49)
    
    patch_embeddings = linear_proj(patches)  # (B, 16, 48)

    # Add positional encoding
    positions = torch.arange(16, device=img_tensor.device)
    rows = positions // 4
    cols = positions % 4
    pos_embed = torch.cat([row_embed(rows), col_embed(cols)], dim=1)  # (16, 48)
    patch_embeddings = patch_embeddings + pos_embed.unsqueeze(0)  # (1, 16, 48)

    return patch_embeddings


# --- Dataset ---
train_data = MultiDigitDataset(data_dir="data/multidigit", split="train")
val_data = MultiDigitDataset(data_dir="data/multidigit", split="val")
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# --- Model ---
encoder = MultiHeadEncoderModel(num_classes=10, dim_k=EMBEDDING_DIM, num_heads=4, num_blocks=6)
decoder = DigitTransformerDecoder(vocab_size=VOCAB_SIZE, dim_model=EMBEDDING_DIM, num_heads=2, num_layers=2, max_len=SEQ_LEN)
model = EncoderDecoderModel(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss(ignore_index=12)  # ignore <pad> token
acc_fn = Accuracy(task="multiclass", num_classes=VOCAB_SIZE, average='micro').to(device)

# --- Training loop ---
def train():
    model.train()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    for epoch in range(EPOCHS):
        total_loss, total_acc, total_tokens = 0, 0, 0
        for batch_imgs, tgt_input, tgt_output in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            batch_imgs = patch_image_tensor(batch_imgs.to(device))
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            optimizer.zero_grad()
            padding_mask = (tgt_input == 12)
            logits = model(batch_imgs, tgt_input, tgt_key_padding_mask=padding_mask)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), tgt_output.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_imgs.size(0)
            preds = logits.argmax(dim=-1)
            mask = tgt_output != 12
            total_acc += ((preds == tgt_output) & mask).sum().item()
            total_tokens += mask.sum().item()

        # Validation
        model.eval()
        val_loss, val_acc, val_tokens = 0, 0, 0
        with torch.no_grad():
            for batch_imgs, tgt_input, tgt_output in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                batch_imgs = patch_image_tensor(batch_imgs.to(device))
                tgt_input = tgt_input.to(device)
                tgt_output = tgt_output.to(device)

                padding_mask = (tgt_input == 12)
                logits = model(batch_imgs, tgt_input, tgt_key_padding_mask=padding_mask)
                loss = loss_fn(logits.view(-1, VOCAB_SIZE), tgt_output.view(-1))
                val_loss += loss.item() * batch_imgs.size(0)
                preds = logits.argmax(dim=-1)
                mask = tgt_output != 12
                val_acc += ((preds == tgt_output) & mask).sum().item()
                val_tokens += mask.sum().item()

            wandb.log({
                "train_loss": total_loss / len(train_data),
                "train_acc": total_acc / total_tokens,
                "val_loss": val_loss / len(val_data),
                "val_acc": val_acc / val_tokens,
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

if __name__ == "__main__":
    train()
