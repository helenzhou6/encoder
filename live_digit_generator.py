import streamlit as st
import random
from PIL import Image
import torch
from torchvision import datasets
from torchvision.transforms import ToPILImage
import os

# --- Config ---
MNIST_PATH = "./raw_data"
CANVAS_SIZE = 96
DISPLAY_SCALE = 4
DIGIT_SIZE = 28
MAX_DIGITS = 4

# Load MNIST dataset
@st.cache_resource
def load_mnist():
    return datasets.MNIST(root=MNIST_PATH, train=False, download=False)

mnist = load_mnist()
to_pil = ToPILImage()

# Compose digits randomly onto a canvas
def generate_multidigit_image():
    num_digits = random.randint(1, MAX_DIGITS)
    indices = [random.randint(0, len(mnist) - 1) for _ in range(num_digits)]
    digits = [mnist[i][0] for i in indices]

    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
    used_boxes = []

    for digit_img in digits:
        placed = False
        attempt = 0
        while not placed and attempt < 50:
            x = random.randint(0, CANVAS_SIZE - DIGIT_SIZE)
            y = random.randint(0, CANVAS_SIZE - DIGIT_SIZE)
            new_box = [x, y, x + DIGIT_SIZE, y + DIGIT_SIZE]

            if all((x2 <= x or x1 >= x + DIGIT_SIZE or y2 <= y or y1 >= y + DIGIT_SIZE)
                   for x1, y1, x2, y2 in used_boxes):
                canvas.paste(digit_img, (x, y))
                used_boxes.append(new_box)
                placed = True
            attempt += 1

    return canvas

# --- Streamlit UI ---
st.markdown("<h1 style='font-size: 24px;'>🎨 Digit Sequence Recognizer</h1>", unsafe_allow_html=True)

if "generated_image" not in st.session_state:
    st.session_state.generated_image = None

if st.button("Generate Digit Image"):
    st.session_state.generated_image = generate_multidigit_image()

if st.session_state.generated_image:
    big_img = st.session_state.generated_image.resize((CANVAS_SIZE * DISPLAY_SCALE, CANVAS_SIZE * DISPLAY_SCALE), Image.NEAREST)
    st.image(big_img, caption="Generated Image")
    st.success("Image generated! You can now use it for inference.")
    use_image = st.button("Use this image for inference")
    if use_image:
        st.write("✅ Image selected for inference.")

        # --- Model Inference ---
        from torchvision import transforms
        import torch
        import numpy as np

        # Preprocess: augment, resize and ToTensor
        tf = transforms.Compose([
            #transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Resize((96, 96)),
            transforms.ToTensor()
        ])
        gray = st.session_state.generated_image.convert('L')
        gray = gray.resize((96, 96), Image.LANCZOS)
        img_tensor = tf(gray).unsqueeze(0)  # (1, 1, 96, 96)

        # Load the model
        from src.model.init_decoder import EncoderDecoderModel, DigitTransformerDecoder
        from src.model.init_encoder import MultiHeadEncoderModel

        VOCAB_SIZE = 13
        SEQ_LEN = 6
        NUM_ENCODER_BLOCKS = 4
        NUM_ENCODER_ATTHEADS = 4
        EMBEDDING_DIM = 24

        encoder = MultiHeadEncoderModel(num_classes=10, dim_k=EMBEDDING_DIM, num_heads=NUM_ENCODER_ATTHEADS, num_blocks=NUM_ENCODER_BLOCKS)
        decoder = DigitTransformerDecoder(vocab_size=VOCAB_SIZE, dim_model=EMBEDDING_DIM, num_heads=NUM_ENCODER_ATTHEADS, num_layers=NUM_ENCODER_BLOCKS, max_len=SEQ_LEN)
        model = EncoderDecoderModel(encoder, decoder)
        model.load_state_dict(torch.load("digit_transformer3.pt", map_location="cpu"))
        model.eval()

        def patch_image_tensor(img_tensor):
            return model.encoder.patch_image_tensor(img_tensor)

        patch_embed = patch_image_tensor(img_tensor)

        # Inference loop
        sos_token = torch.tensor([[10]])
        preds = []
        for _ in range(SEQ_LEN):
            tgt_input = torch.tensor([[10] + preds])
            padding_mask = (tgt_input == 12)
            logits = model(patch_embed, tgt_input, tgt_key_padding_mask=padding_mask)
            next_token = logits.argmax(dim=-1)[:, -1].item()
            preds.append(next_token)
            if next_token == 11:
                break
        preds += [12] * (SEQ_LEN - len(preds))

        # Display predictions
        st.markdown("<h2 style='font-size: 28px;'>Prediction</h2>", unsafe_allow_html=True)
        st.text(f"Raw tokens: {preds}")
        digits_only = [x for x in preds if 0 <= x <= 9]
        digits_str = ' '.join(map(str, digits_only))
        st.markdown(f"<div style='font-size: 48px; font-weight: bold; color: #2c3e50; text-align: center;'>{digits_str}</div>", unsafe_allow_html=True)