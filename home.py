import streamlit as st
from PIL import Image

def app(_,s_state):
    #s_state.title = 'Home'
    st.image('./LOGO-QSAR-lit.tif')
    
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
    - App built in `Python` + `Streamlit` by [José Teófilo Moreira Filho](http://lattes.cnpq.br/3464351249761623), Henric P. V. Gil, Rodolpho C. Braga, Vinicius M. Alves , Bruno J. Neves, and Carolina H. Andrade.
    - Plese follow our research:
        [Lattes](http://lattes.cnpq.br/3464351249761623),
        [ResearchGate](https://www.researchgate.net/profile/Jose-Teofilo-Filho),
        [Google Scholar](https://scholar.google.com/citations?user=0I1GiOsAAAAJ&hl=pt-BR),
        [ORCID](https://orcid.org/0000-0002-0777-280X).
    
    </p>
    """,unsafe_allow_html=True)
    #Page description and credits
    #image = Image.open('logo.png')
