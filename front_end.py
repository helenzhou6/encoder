import ssl, certifi
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import transforms
from PIL import Image

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# --- Load Transformer Model ---
@st.cache_resource
def load_model():
    from src.model.init_decoder import EncoderDecoderModel
    from src.model.init_encoder import MultiHeadEncoderModel
    from src.model.init_decoder import DigitTransformerDecoder

    # Match training config
    EMBEDDING_DIM = 96
    NUM_ENCODER_BLOCKS = 4
    NUM_ATTENTION_HEADS = 4
    VOCAB_SIZE = 13
    SEQ_LEN = 6

    encoder = MultiHeadEncoderModel(num_classes=10, dim_k=EMBEDDING_DIM,
                                     num_heads=NUM_ATTENTION_HEADS, num_blocks=NUM_ENCODER_BLOCKS)
    decoder = DigitTransformerDecoder(vocab_size=VOCAB_SIZE, dim_model=EMBEDDING_DIM,
                                      num_heads=NUM_ATTENTION_HEADS, num_layers=NUM_ENCODER_BLOCKS,
                                      max_len=SEQ_LEN)
    model = EncoderDecoderModel(encoder, decoder)
    model.load_state_dict(torch.load("digit_transformer.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# --- Patch embedding components ---
PATCH_SIZE = 16
EMBEDDING_DIM = 96
linear_proj = torch.nn.Linear(PATCH_SIZE * PATCH_SIZE, EMBEDDING_DIM)
row_embed = torch.nn.Embedding(6, EMBEDDING_DIM // 2)
col_embed = torch.nn.Embedding(6, EMBEDDING_DIM // 2)

@torch.no_grad()
def patch_image_tensor(img_tensor):
    patches = img_tensor.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)  # (B, 1, 6, 6, 16, 16)
    patches = patches.contiguous().view(img_tensor.size(0), 36, -1)  # (B, 36, 256)
    patch_embeddings = linear_proj(patches)  # (B, 36, 96)

    # Add positional encoding
    positions = torch.arange(36, device=img_tensor.device)
    rows = positions // 6
    cols = positions % 6
    pos_embed = torch.cat([row_embed(rows), col_embed(cols)], dim=1)  # (36, 96)
    patch_embeddings = patch_embeddings + pos_embed.unsqueeze(0)  # (1, 36, 96)

    return patch_embeddings

# --- Preprocessing ---
def preprocess_canvas_image(img_data):
    rgb = img_data[:, :, :3].astype('uint8')
    gray = Image.fromarray(rgb).convert('L')
    gray = gray.resize((96, 96), Image.LANCZOS)
    tf = transforms.Compose([
        transforms.ToTensor(),
    ])
    return tf(gray)

# --- Streamlit UI ---
st.title("🎨 Digit Sequence Recognizer (Transformer)")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=2.5,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=96,
    height=96,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Classify"):
    if canvas_result.image_data is None:
        st.warning("Please draw something first!")
    else:
        img_tensor = preprocess_canvas_image(canvas_result.image_data).unsqueeze(0)  # (1, 1, 96, 96)
        patch_embed = patch_image_tensor(img_tensor)

        sos_token = torch.tensor([[10]])  # <sos>
        preds = []
        for _ in range(6):
            tgt_input = torch.tensor([ [10] + preds ])  # always start with <sos>
            padding_mask = (tgt_input == 12)
            logits = model(patch_embed, tgt_input, tgt_key_padding_mask=padding_mask)
            next_token = logits.argmax(dim=-1)[:, -1].item()
            preds.append(next_token)
            if next_token == 11:  # <eos>
                break

        # Display
        st.subheader("Prediction")
        st.text(f"Raw tokens: {preds}")
        digits_only = [x for x in preds if 0 <= x <= 9]
        st.write(f"**Digits:** {' '.join(map(str, digits_only))}")

if st.button("New Drawing"):
    st.rerun()