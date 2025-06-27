import ssl, certifi
import wandb
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import transforms
from PIL import Image
from model.utils import init_wandb, load_artifact_path

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# --- Load Transformer Model ---
wandb.init(
        entity="week3-rebels",
        project="digit-transformer"
    )

#model_path = load_artifact_path("digit_transformer")
model_path = "digit_transformer3.pt"

NUM_PATCHES = 576 #36
NUM_CUTS = 24 #6

@st.cache_resource
def load_model():
    from src.model.init_decoder import EncoderDecoderModel
    from src.model.init_encoder import MultiHeadEncoderModel
    from src.model.init_decoder import DigitTransformerDecoder

    # Match training config
    EMBEDDING_DIM = 24 #96
    NUM_ENCODER_BLOCKS = 4 #6
    NUM_ATTENTION_HEADS = 4
    VOCAB_SIZE = 13
    SEQ_LEN = 6

    encoder = MultiHeadEncoderModel(num_classes=10, dim_k=EMBEDDING_DIM,
                                     num_heads=NUM_ATTENTION_HEADS, num_blocks=NUM_ENCODER_BLOCKS)
    decoder = DigitTransformerDecoder(vocab_size=VOCAB_SIZE, dim_model=EMBEDDING_DIM,
                                      num_heads=NUM_ATTENTION_HEADS, num_layers=NUM_ENCODER_BLOCKS,
                                      max_len=SEQ_LEN)
    model = EncoderDecoderModel(encoder, decoder)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

with torch.no_grad():
    model = load_model()

with st.expander("📚 How the model works"):
    st.markdown("""
    This model uses a Transformer-based encoder-decoder architecture.

    - The **encoder** splits the input image into patches and processes them using multi-head self-attention.
    - The **decoder** generates a digit sequence, starting from a <sos> (start of sequence) token, one token at a time.

    The output is a fixed-length sequence of up to 4 digits, ending with an <eos> (end of sequence) token.
    """)
    st.image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformer_architecture.png", use_column_width=True)

# --- Preprocessing ---
def preprocess_canvas_image(img_data):
    rgb = img_data[:, :, :3].astype('uint8')
    gray = Image.fromarray(rgb).convert('L')
    gray = gray.resize((96, 96), Image.LANCZOS)
    tf = transforms.Compose([
        transforms.ToTensor()
    ])
    return tf(gray)

# --- Streamlit UI ---
st.title("Digit Transformer ✨")
st.markdown("A mini transformer model for recognizing sequences of handwritten digits.")

st.markdown("""
    <style>
    div[data-testid="stCanvas"] canvas {
        transform: scale(4);
        transform-origin: top left;
        image-rendering: pixelated;
    }
    </style>
""", unsafe_allow_html=True)

DISPLAY_SCALE = 4  # display 4x larger

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🖍️ Draw a multi-digit image")
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=96 * DISPLAY_SCALE,
        height=96 * DISPLAY_SCALE,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True,
    )
    if st.button("🧹 Clear Canvas"):
        st.rerun()

with col2:
    if st.button("Classify"):
        if canvas_result.image_data is None:
            st.warning("Please draw something first!")
        else:
            img_tensor = preprocess_canvas_image(canvas_result.image_data).unsqueeze(0)  # (1, 1, 96, 96)
            patch_embed = model.encoder.patch_image_tensor(img_tensor)

            sos_token = torch.tensor([[10]])  # <sos>
            preds = []
            SEQ_LEN = 6
            for _ in range(SEQ_LEN):
                tgt_input = torch.tensor([ [10] + preds ])  # always start with <sos>
                padding_mask = (tgt_input == 12)
                logits = model(patch_embed, tgt_input, tgt_key_padding_mask=padding_mask)
                next_token = logits.argmax(dim=-1)[:, -1].item()
                preds.append(next_token)
                if next_token == 11:  # <eos>
                    break
            preds += [12] * (SEQ_LEN - len(preds))  # pad to length

            # Display
            st.subheader("🔍 Prediction")
            st.markdown(f"<div style='font-size: 24px; color: gray;'>Raw tokens: {preds}</div>", unsafe_allow_html=True)
            digits_only = [x for x in preds if 0 <= x <= 9]
            st.markdown(f"<div style='font-size: 48px; font-weight: bold; color: #4CAF50;'>Digits: {' '.join(map(str, digits_only))}</div>", unsafe_allow_html=True)