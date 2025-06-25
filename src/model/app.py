import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from PIL import Image

from inference import predict_digit


st.title("Digit Classifier 🤖")
st.markdown(
    """ 
    Digit Classifier 🤖 will try and predict the digit that you've drawn. :rainbow[**WOWSIES**]

    Do your best drawing of a digit between 0-9 and hit 'Classify'
    """
)

canvas_result = st_canvas(
    stroke_width=20,
    stroke_color="#fff",
    background_color="#000",
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if 'text' not in st.session_state:
    st.session_state['text'] = "🤖 patiently awaits your beautiful drawing of a digit..."


def on_button_click():
    if canvas_result.image_data is not None:

        image = Image.fromarray(canvas_result.image_data.astype("uint8"))  # Convert to PIL image
        image = image.convert("L")  # Convert to grayscale

        result = predict_digit(image)
        if result:
            st.session_state['text'] = f"Prediction made! 🤖 thinks it's {result}"
            st.session_state['predicted_digit'] = result
        else:
            st.session_state['text'] = f"Something went wrong, see logs.. {result}"

st.button(label="Classify", type="primary", on_click=on_button_click)

st.write(st.session_state['text'])
