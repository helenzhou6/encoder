import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
import io
from PIL import Image

# Fix random seed for reproducibility
torch.manual_seed(42)

def compute_attention(Q, K, scale=True):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1)
    if scale:
        scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention = F.softmax(scores, dim=-1)
    return scores.detach().numpy().flatten(), attention.detach().numpy()

def plot_distribution_and_heatmap(scores, attention, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Distribution
    sns.histplot(scores, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title(f"{title_prefix} Distribution\nμ={np.mean(scores):.2f}, σ={np.std(scores):.2f}")

    # Attention heatmap
    sns.heatmap(attention[0], ax=axes[1], cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title(f"{title_prefix} Attention Heatmap")
    axes[1].set_xlabel("Key")
    axes[1].set_ylabel("Qry")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    wandb.log({f"{title_prefix}": wandb.Image(image)})    
    plt.close(fig)

def inspect(dim, epochs=5):
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1} for d_k = {dim} ---")
        Q = torch.randn(1, 16, dim)
        K = torch.randn(1, 16, dim)

        raw_scores, attention_no_scale = compute_attention(Q, K, scale=False)
        plot_distribution_and_heatmap(raw_scores, attention_no_scale, f"Epoch {epoch+1} - No Scaling (d_k={dim})")

        scaled_scores, attention_scaled = compute_attention(Q, K, scale=True)
        plot_distribution_and_heatmap(scaled_scores, attention_scaled, f"Epoch {epoch+1} - Scaled by √d_k (d_k={dim})")

if __name__ == "__main__":
    wandb.init(project="attention_inspection", name="embedding_dim_scaling", config={"embedding_dims": [4, 16, 49, 64, 128], "epochs": 5})
    for dim in [4, 16, 49, 64, 128]:
        inspect(dim, epochs=5)
    wandb.finish()