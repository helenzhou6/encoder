import streamlit as st
from streamlit_drawable_canvas import st_canvas

from call_api import predict_digit_api

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
        result = predict_digit_api(canvas_result.image_data)
        if result.values() and len(result.values()) == 2:
            predicted_digit = result.values()
            st.session_state['text'] = f"Prediction made! 🤖 thinks it's {predicted_digit} and with {conf_percent:.1f}% confidence"
            st.session_state['predicted_digit'] = predicted_digit
        else:
            st.session_state['text'] = f"Something went wrong, see logs.. {result}"

st.button(label="Classify", type="primary", on_click=on_button_click)

st.write(st.session_state['text'])
