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

import base64
import functools
from io import BytesIO
import os
import warnings

from st_aggrid import AgGrid
warnings.filterwarnings(action='ignore')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, RDPaths
from rdkit.Chem import AllChem,  DataStructs, Draw, rdBase, rdCoordGen, rdDepictor
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D, SimilarityMaps
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MACCSkeys
#print(f'RDKit: {rdBase.rdkitVersion}')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

import streamlit as st

import joblib
from rdkit.Chem import MACCSkeys
from skopt import BayesSearchCV

import pickle

sns.set_style("whitegrid")

def app(df):
    ########################################################################################################################################
    # Function Settings
    ########################################################################################################################################

    # from chainer-chemistry's visualizer_utils.py
    def red_blue_cmap(x):
        """Red to Blue color map
        Args:
            x (float): value between -1 ~ 1, represents normalized saliency score
        Returns (tuple): tuple of 3 float values representing R, G, B.
        """
        if x > 0:
            # Red for positive value
            # x=0 -> 1, 1, 1 (white)
            # x=1 -> 1, 0, 0 (red)
            return 1.0, 1.0 - x, 1.0 - x
        else:
            # Blue for negative value
            x *= -1
            return 1.0 - x, 1.0 - x, 1.0

    def is_visible(begin, end):
        if begin <= 0 or end <= 0:
            return 0
        elif begin >= 1 or end >= 1:
            return 1
        else:
            return (begin + end) * 0.5

    def color_bond(bond, saliency, color_fn):
        begin = saliency[bond.GetBeginAtomIdx()]
        end = saliency[bond.GetEndAtomIdx()]
        return color_fn(is_visible(begin, end))


    def moltopng(mol, atom_colors, bond_colors, molSize=(450,150),kekulize=True):
        mc = Chem.Mol(mol.ToBinary())
        if kekulize:
            try:
                Chem.Kekulize(mc)
            except:
                mc = Chem.Mol(mol.ToBinary())
        if not mc.GetNumConformers():
            rdDepictor.Compute2DCoords(mc)
        drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0],molSize[1])
        drawer.drawOptions().useBWAtomPalette()
        drawer.drawOptions().padding = .2
        drawer.DrawMolecule(
            mc,
            highlightAtoms=[i for i in range(len(atom_colors))],
            highlightAtomColors=atom_colors,
            highlightBonds=[i for i in range(len(bond_colors))],
            highlightBondColors=bond_colors,
            highlightAtomRadii={i: .5 for i in range(len(atom_colors))}
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    def label_cat(label):
        return '$+$' if bool(label!=0) else '$\cdot$'

    # https://iwatobipen.wordpress.com/2017/02/25/draw-molecule-with-atom-index-in-rdkit/
    def mol_with_atom_index(mol):
        import copy
        atoms = mol.GetNumAtoms()
        molwithidx = copy.deepcopy(mol)
        for idx in range( atoms ):
            molwithidx.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( molwithidx.GetAtomWithIdx( idx ).GetIdx() ) )
        return molwithidx

    def draw_mols_with_idx(mol, atom_colors, bond_colors):
        images = []
        _img = moltopng(mol, atom_colors, bond_colors, molSize=(720, 480))
        output = BytesIO()
        output.write(_img)
        encoded_string = base64.b64encode(output.getvalue()).decode()
        images.append(f"<img style='width: 720px; margin: 0px; float: left; border: 0px solid black;' src='data:_img/png;base64,{encoded_string}' />")
        _img = moltopng(mol_with_atom_index(mol), {}, {}, molSize=(720, 470))
        output = BytesIO()
        output.write(_img)
        encoded_string = base64.b64encode(output.getvalue()).decode()
        images.append(f"<img style='width: 720px; margin: 0px; float: left; border: 0px solid black;' src='data:_img/png;base64,{encoded_string}' />")
        images = ''.join(images)
    #    display(HTML(images))
        st.write(images, unsafe_allow_html=True,)

    def plot_explainable_images(mol: rdkit.Chem.Mol, weight_fn, weights=None, atoms=['C', 'N', 'O', 'S', 'F', 'Cl', 'P', 'Br']):
        symbols = [f'{mol.GetAtomWithIdx(i).GetSymbol()}_{i}' for i in range(mol.GetNumAtoms())]
        df = pd.DataFrame(columns=atoms)
        if weights is None:
            contribs = weight_fn(mol)
        else:
            contribs = weights
        num_atoms = mol.GetNumAtoms()
        arr = np.zeros((num_atoms, len(atoms)))
        for i in range(mol.GetNumAtoms()):
            _a = mol.GetAtomWithIdx(i).GetSymbol()
            arr[i,atoms.index(_a)] = contribs[i]
        df = pd.DataFrame(arr, index=symbols, columns=atoms)
        weights, vmax = SimilarityMaps.GetStandardizedWeights(contribs)
        vmin = -vmax
        atom_colors = {i: red_blue_cmap(e) for i, e in enumerate(weights)}
        # bondlist = [bond.GetIdx() for bond in mol.GetBonds()]
        bond_colors = {i: color_bond(bond, weights, red_blue_cmap) for i, bond in enumerate(mol.GetBonds())}
        draw_mols_with_idx(mol, atom_colors, bond_colors)

        fig = plt.figure(figsize=(18, 9))
        grid = plt.GridSpec(15, 10)
        ax = fig.add_subplot(grid[1:, -1])

        ax.barh(range(mol.GetNumAtoms()), np.maximum(0, df.values).sum(axis=1), color='C3')
        ax.barh(range(mol.GetNumAtoms()), np.minimum(0, df.values).sum(axis=1), color='C0')

        ax.set_yticks(range(mol.GetNumAtoms()))
        ax.set_ylim(-.5, mol.GetNumAtoms()-.5)

        ax.axvline(0, color='k', linestyle='-', linewidth=.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.tick_params(axis='both', which='both', left=False, labelleft=False)

        ax = fig.add_subplot(grid[1:, :-1], sharey=ax)
        im = ax.imshow(df.values, cmap='bwr', vmin=vmin, vmax=vmax, aspect='auto')

        ax.set_yticks(range(mol.GetNumAtoms()))
        ax.set_ylim(mol.GetNumAtoms() -.5, -.5)

        symbols= {i: f'${mol.GetAtomWithIdx(i).GetSymbol()}_{{{i}}}$' for i in range(mol.GetNumAtoms())}

        ax.set_yticklabels(symbols.values())
        ax.set_xticks(range(len(atoms)))
        ax.set_xlim(-.5, len(atoms) -.5)
        ax.set_xticklabels(atoms, rotation=90)
        ax.set_ylabel('Node')

        for (j,i),label in np.ndenumerate(df.values):
            ax.text(i,j, label_cat(label) ,ha='center',va='center')
        ax.tick_params(axis='both', which='both', bottom=True, labelbottom=True, top=False, labeltop=False)
        ax.grid(b=None)

        ax = fig.add_subplot(grid[0, :-1])
        fig.colorbar(mappable=im, cax=ax, orientation='horizontal')
        ax.tick_params(axis='both', which='both', bottom=False, labelbottom=False, top=True, labeltop=True)

        st.write(fig, unsafe_allow_html=True,)

    # ## Fingerprint
    # https://iwatobipen.wordpress.com/2018/11/07/visualize-important-features-of-machine-leaning-rdkit/
    def mol2fp(mol,radius=2, nBits=2048):
        bitInfo={}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=bitInfo)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr, bitInfo

    # ## GetAtomicWeightsForModel
    def get_proba(fp, proba_fn, class_id):
        return proba_fn((fp,))[0][class_id]
    # codigo seu mr teofilo ??? kakakkdasd
    # デフォルトでGetMorganFingerprintの引数が2048ビットになっているため
    def fp_partial(nBits, radius):
        return functools.partial(SimilarityMaps.GetMorganFingerprint, nBits=nBits, radius=radius)

    def rdkit_numpy_convert(fp):
        output = []
        for f in fp:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(f, arr)
            output.append(arr)
        return np.asarray(output)

    ########################################################################################################################################
    # Page Title
    ########################################################################################################################################

    ########################################################################################################################################
    # Sidebar - Upload File and select columns
    ########################################################################################################################################

    # Upload File
    #repeated code
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
    [Example CSV input file](https://github.com/joseteofilo/data_qsarlit/blob/master/example_data_for_probability_maps.csv)
    """)


    # Select columns
    #repeated code
    with st.sidebar.header('2. Enter column names for SMILES and Compound id'):
        name_smiles = st.sidebar.text_input('Enter column name for SMILES', 'SMILES')
        column_id = st.sidebar.text_input('Enter column name with the ID of compounds', 'ID')
    with st.sidebar.expander("Dont have compound id"):
            st.write("If you don´t have a column with the compounds id´s ***click*** the button and use standard indexes")
            no_compound_id=st.button("Use standard indexes")

    #st.header('**Original input data**')

    # Read Uploaded file and convert to pandas
    #repeated code
    if uploaded_file is not None:
        # Read CSV data
        df = pd.read_csv(uploaded_file, sep=',')
        df =pd.DataFrame(df)
        st.header('**Original input data**')

        AgGrid(df)

    else:
        st.info('Awaiting for CSV file to be uploaded.')

    ########################################################################################################################################
    # Sidebar - Select descriptor
    ########################################################################################################################################
    with st.sidebar.header('Morgan descriptor'):

        radius_input = st.sidebar.number_input('Enter the radius', min_value=None, max_value=None, value=int(2))
        nbits_input = st.sidebar.number_input('Enter the number of bits', min_value=None, max_value=None, value=int(2048))
    ########################################################################################################################################
    # Upload models
    ########################################################################################################################################
    # Upload File
    #repeated code
    with st.sidebar.header('4. Upload your model'):
        model_file = st.sidebar.file_uploader("Upload your model in PKL file", type=["pkl"])
    st.sidebar.markdown("""
    [Example PKL input file](https://github.com/joseteofilo/data_qsarlit/blob/master/model_rf_morgan_r2_2048bits.pkl)
    """)
    #repeated code
    def load_model(model):
        loaded_model = joblib.load(model)
        return loaded_model

    ########################################################################################################################################
    # Select compound
    ########################################################################################################################################
    #repeated code
    if uploaded_file is not None:
        if no_compound_id: 
            ids=df.index
        elif column_id is not None:
            try:
                ids=df[column_id]
            except KeyError:
                ids=df.index
        st.header('**Interpret predictions**')
        selected_name = st.selectbox('Select by ID', ids)
        #st.header(f'Structure:')
        #st.image(moltoimage(df.loc[df[column_id] == selected_name][name_smiles].values[0]))
        mol = df.loc[ids == selected_name,name_smiles].values[0]
        mol = Chem.MolFromSmiles(mol)

        if model_file is not None:
            clf = load_model(model_file)

    ########################################################################################################################################
    # Maps
    ########################################################################################################################################
    if st.sidebar.button('Generate maps'):
        #show_pred_results(mol, clf)
        weights = SimilarityMaps.GetAtomicWeightsForModel(mol, fp_partial(nbits_input, radius_input), lambda x: get_proba(x, clf.predict_proba,1))

        #st.write(weights, unsafe_allow_html=True)
        st.header('Contribution Maps')

        plot_explainable_images(mol, weight_fn=None, weights=weights, atoms=['C', 'N', 'O', 'S', 'F', 'Cl', 'P', 'Br'])
