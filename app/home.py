import os
import streamlit as st
from PIL import Image

def app(_, s_state):
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the image file
    image_path = os.path.join(current_dir, 'LOGO-QSAR-lit.tif')

    # Display the image
    st.image(image_path)

    st.markdown("""
    <style>
    p {
    margin-top: 2%;
    }
    </style>
    <p>
    This app allows you to curate data, calculate molecular descriptors, develop Machine Learning models, perform virtual screening,
    and interpret the predictions with probability maps for computational toxicology and drug discovery projects.

    **Credits**
    - App built in `Python` + `Streamlit` by José Teófilo Moreira Filho (<a href="http://lattes.cnpq.br/3464351249761623" target="_blank">Lattes</a>, <a href="https://scholar.google.com/citations?user=0I1GiOsAAAAJ&hl=pt-BR" target="_blank">Google Scholar</a> ,<a href="https://orcid.org/0000-0002-0777-280X" target="_blank">ORCID</a>), <a href="https://rcbraga.github.io/" target="_blank">Rodolpho C. Braga</a>, Henric P. V. Gil, Vinicius M. Alves , Bruno J. Neves, and <a href="http://labmol.com.br/" target="_blank">Carolina H. Andrade</a>.
    </p>
    """, unsafe_allow_html=True)
