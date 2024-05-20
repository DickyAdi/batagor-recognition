import streamlit as st
from scripts import predict
from tempfile import NamedTemporaryFile
import os

st.header('Predict your image!')
st.subheader('Is it recognized as a batagor or not?')

image = st.file_uploader("Upload you images!", type=['png', 'jpg', '.jpeg'], accept_multiple_files=False)
if image:

    with NamedTemporaryFile(delete=False) as tmp_files:
        tmp_files.write(image.read())
        tmp_files_path = tmp_files.name
    try:    
        res = predict.get_prediction(tmp_files_path).squeeze()
        if res > 0:
            st.write('It seems like your image is recognized as a Batagor!!')
        else:
            st.write('MEHHH!! NOT A BATAGOR')
    finally:
        os.remove(tmp_files_path)