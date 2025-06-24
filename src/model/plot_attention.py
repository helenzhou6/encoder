
import matplotlib.pyplot as plt
import wandb
import os

def visualise_attention(normalised_attention, step):
    attention_matrix = normalised_attention[0]
    if attention_matrix.ndim == 3:
        attention_matrix = attention_matrix[0]
    attention_matrix = attention_matrix.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.matshow(attention_matrix, cmap='viridis', fignum=1)
    plt.title("Attention map")
    plt.xlabel("One Key patch")
    plt.ylabel("One Query patch") 
    plt.colorbar()

    # Save plot locally
    plot_path = "attention_map.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    # Log to wandb
    wandb.log({"attention_map": wandb.Image(plot_path)}, step=step)

    # Optionally: also persist the image file in the W&B run directory
    wandb.save(plot_path)

    # Clean up the local file (optional)
    if os.path.exists(plot_path):
        os.remove(plot_path)