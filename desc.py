########################################################################################################################################
# Credits
########################################################################################################################################

# Developed by José Teófilo Moreira Filho, Ph.D.
# teofarma1@gmail.com
# http://lattes.cnpq.br/3464351249761623
# https://www.researchgate.net/profile/Jose-Teofilo-Filho
# https://scholar.google.com/citations?user=0I1GiOsAAAAJ&hl=pt-BR
# https://orcid.org/0000-0002-0777-280X

########################################################################################################################################
# Importing packages
########################################################################################################################################

from st_aggrid import AgGrid
import streamlit as st

import base64
import functools
from io import BytesIO
import os
import warnings
warnings.filterwarnings(action='ignore')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

from rdkit import Chem, DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import PandasTools
from rdkit.Chem import MACCSkeys

import numpy as np
from numpy import sqrt
from numpy import argmax

import pandas as pd
from pandas import DataFrame
import utils
from sklearn.preprocessing import StandardScaler

def app(df, s_state):
    cc = utils.Custom_Components()
    ########################################################################################################################################
    # Functions
    ########################################################################################################################################
    def persist_dataframe(updated_df,col_to_delete):
            # drop column from dataframe
            delete_col = st.session_state[col_to_delete]

            if delete_col in st.session_state[updated_df]:
                st.session_state[updated_df] = st.session_state[updated_df].drop(columns=[delete_col])
            else:
                st.sidebar.warning("Column previously deleted. Select another column.")
            with st.container():
                st.header("**Updated input data**") 
                AgGrid(st.session_state[updated_df])
                st.header('**Original input data**')
                AgGrid(df)

    def rdkit_numpy_convert(fp):
        output = []
        for f in fp:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(f, arr)
            output.append(arr)
        return np.asarray(output)

    def smiles_to_fp(mols, radius, nbits, method):
        # convert smiles to RDKit mol object
        fps = []
        for mol in mols:
            mol = Chem.MolFromSmiles(mol)
            if method == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
                fps.append(fp)

            if method == 'featmorgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits, useFeatures=True)
                fps.append(fp)

        return fps

    def smiles_to_fp_maccs(mols, method):
        # convert smiles to RDKit mol object
        fps = []
        for mol in mols:
            mol = Chem.MolFromSmiles(mol)

            if method == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(mol)
                fps.append(fp)

        return fps

    ########################################################################################################################################
    # Sidebar - Upload File and select columns
    ########################################################################################################################################

    # Upload File
    

    # Read Uploaded file and convert to pandas
    if df is not None:
        # Read CSV data
        #df = pd.read_csv(uploaded_file, sep=',')
    #st.sidebar.write('---')

    # Select columns
        with st.sidebar.header('1. Enter column names'):
            name_smiles = st.sidebar.selectbox('Enter column name for SMILES', df.columns)
            #name_activity = st.sidebar.text_input('Enter column name for Activity (Active and Inactive should be 1 and 0, respectively)', 'Outcome')

        #st.header('**Original input data**')


        ########################################################################################################################################
        # Sidebar - Select descriptor
        ########################################################################################################################################

        with st.sidebar.header('2. Calculate Descriptor'):
            # Select fingerprint method
            with st.sidebar.header('Morgan parameters'):
                radius = st.sidebar.number_input('Enter the radius', min_value=1, max_value=10, value=int(2))
                nbits = st.sidebar.number_input('Enter the number of bits', min_value=1024, max_value=None, value=int(2048))

        ########################################################################################################################################
        # Apply descriptor calc
        ########################################################################################################################################

        if st.sidebar.button('Calculate descriptors'):
            df_smiles = df
            # Calculate selected descriptor
            for i in range(nbits):
                df_smiles[f"{i}"] = 0
            descriptor = smiles_to_fp(mols=df_smiles[name_smiles], radius=radius, nbits=nbits, method="morgan")
            #st.write(descriptor)
            for i in range(df.shape[0]):
                for j in range(nbits):
                    if s_state.title != "Calculate Descriptors":
                        break
                    df_smiles.loc[j,f"{i}"] = descriptor[i][j]
                

            # Convert descritor to Pandas dataframe
            #df_smiles["descriptor"] = rdkit_numpy_convert(descriptor)

            # Join calculated descriptor with activity column
            #df_descritor = df_descritor.join(s_state.df.drop(name_smiles, axis=1))

            # Print descriptors
            
            cc.AgGrid(df_smiles,"Calculated descriptors")
        ########################################################################################################################################
        # Descriptor download
        ########################################################################################################################################

        # File download
            def filedownload(df):
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
                st.header("**Download descriptors**")
                href = f'<a href="data:file/csv;base64,{b64}" download="descriptor_morgan_{radius}_{nbits}.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

            filedownload(df_smiles)
