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

import streamlit as st

import base64
import warnings
warnings.filterwarnings(action='ignore')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

import pandas as pd

from rdkit.Chem import PandasTools
from rdkit import Chem
import utils
from rdkit.Chem.MolStandardize import rdMolStandardize

from st_aggrid import AgGrid
def app(df,s_state):
    #custom = cur.Custom_Components()
    ########################################################################################################################################
    # Functions
    ########################################################################################################################################
    cc = utils.Custom_Components()
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

    def filedownload(df,data):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            st.header(f"**Download {data} data**")
            href = f'<a href="data:file/csv;base64,{b64}" download="{data}_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    def remove_invalid(df,smiles_col):
        for i in df.index:
            try:
                smiles = df[smiles_col][i]
                m = Chem.MolFromSmiles(smiles)
            except:
                df.drop(i, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    # ##################################################################
    # def remove_metals(df):
    #     badAtoms = Chem.MolFromSmarts('[!$([#1,#3,#11,#19,#4,#12,#20,#5,#6,#14,#7,#15,#8,#16,#9,#17,#35,#53])]')
    #     mols = []
    #     for i in df.index:
    #         smiles = df[name_smiles][i]
    #         m = Chem.MolFromSmiles(smiles,)
    #         if m.HasSubstructMatch(badAtoms):
    #             df.drop(i, inplace=True)
    #     df.reset_index(drop=True, inplace=True)
    #     return df
    # ##################################################################
    # def normalize_groups(df):
    #     mols = []
    #     for smi in df[name_smiles]:
    #         m = Chem.MolFromSmiles(smi,sanitize=True,)
    #         m2 = rdMolStandardize.Normalize(m)
    #         smi = Chem.MolToSmiles(m2,kekuleSmiles=True)
    #         mols.append(smi)
    #     norm = pd.DataFrame(mols, columns=["normalized_smiles"])
    #     df_normalized = df.join(norm)
    #     return df_normalized
    # ##################################################################
    # def neutralize(df):
    #     uncharger = rdMolStandardize.Uncharger()
    #     mols = []
    #     for smi in df['normalized_smiles']:
    #         m = Chem.MolFromSmiles(smi,sanitize=True,)
    #         m2 = uncharger.uncharge(m)
    #         smi = Chem.MolToSmiles(m2,kekuleSmiles=True)
    #         mols.append(smi)
    #     neutral = pd.DataFrame(mols, columns=["neutralized_smiles"])
    #     df_neutral = df.join(neutral)
    #     return df_neutral
    # ##################################################################
    # def no_mixture(df):
    #     mols = []
    #     for smi in df["neutralized_smiles"]:
    #         m = Chem.MolFromSmiles(smi,sanitize=True,)
    #         m2 = rdMolStandardize.FragmentParent(m)
    #         smi = Chem.MolToSmiles(m2,kekuleSmiles=True)
    #         mols.append(smi)
    #     no_mixture = pd.DataFrame(mols, columns=["no_mixture_smiles"])
    #     df_no_mixture = df.join(no_mixture)
    #     return df_no_mixture
    # ##################################################################
    # def canonical_tautomer(df):
    #     te = rdMolStandardize.TautomerEnumerator()
    #     mols = []
    #     for smi in df["no_mixture_smiles"]:
    #         m = Chem.MolFromSmiles(smi,sanitize=True,)
    #         m2 = te.Canonicalize(m)
    #         smi = Chem.MolToSmiles(m2,kekuleSmiles=True)
    #         mols.append(smi)
    #     canonical_tautomer = pd.DataFrame(mols, columns=["canonical_tautomer"])
    #     df_canonical_tautomer = df.join(canonical_tautomer)
    #     return df_canonical_tautomer
    # ##################################################################
    # def smi_to_inchikey(df):
    #     inchi = []
    #     for smi in df["canonical_tautomer"]:
    #         m = Chem.MolFromSmiles(smi,sanitize=True,)
    #         m2 = Chem.inchi.MolToInchiKey(m)
    #         inchi.append(m2)
    #     inchikey = pd.DataFrame(inchi, columns=["inchikey"])
    #     df_inchikey = df.join(inchikey)
    #     return df_inchikey
    # ##################################################################
    def file_dependent_code(df):

        # Select columns
        with st.sidebar.header('1. Select column names'):
            name_smiles = st.sidebar.selectbox('Select column containing SMILES', options=df.columns, key="smiles_column")
            name_activity = st.sidebar.selectbox(
                'Select column containing Activity (Active and Inactive should be 1 and 0, respectively or numerical values)', 
                options=df.columns, key="outcome_column"
                )
        curate = utils.Curation(name_smiles)
        st.sidebar.write('---')

        ########################################################################################################################################
        # Sidebar - Select visual inspection
        ########################################################################################################################################

        st.sidebar.header('2. Visual inspection')

        st.sidebar.subheader('Select step for visual inspection')
                
        container = st.sidebar.container()
        _all = st.sidebar.checkbox("Select all")
        data_type = ["Continuous", "Categorical"]
        radio = st.sidebar.radio(
        "Continuous or categorical activity?",
        data_type, key="activity_type",horizontal=True 
        )
        
        options=['Normalization',
                'Neutralization',
                'Mixture_removal',
                'Canonical_tautomers',
                'Chembl_Standardization',
                ]
        if _all:
            selected_options = container.multiselect("Select one or more options:", options, options)
        else:
            selected_options =  container.multiselect("Select one or more options:", options)


        ########################################################################################################################################
        # Apply standardization
        ########################################################################################################################################

        if st.sidebar.button('Standardize'):

            #---------------------------------------------------------------------------------#
            # Remove invalid smiles
            remove_invalid(df,name_smiles)
            df[name_smiles] = curate.smiles_preparator(df[name_smiles])
            st.header("1. Invalid SMILES removed")
            cc.AgGrid(df,key = "invalid_smiles_removed")
            #---------------------------------------------------------------------------------#
            # Remove compounds with metals
            #df = curate.remove_Salt_From_DF(df)
            df = curate.remove_metal(df, name_smiles)
            
            #---------------------------------------------------------------------------------#
            # Normalize groups

            if options[0] in selected_options:

                st.header('**Normalized Groups**')

                normalized = curate.normalize_groups(df)

                # #Generate Image from original SMILES
                # PandasTools.AddMoleculeColumnToFrame(normalized, smilesCol = curate.smiles,
                # molCol = 'Original', includeFingerprints = False)
                # #Generate Image from normalized SMILES
                # PandasTools.AddMoleculeColumnToFrame(normalized, smilesCol = curate.curated_smiles,
                # molCol = 'Normalized', includeFingerprints = False)
                # # Filter only columns containing images
                # normalized_fig = normalized.filter(items = ['Original', "Normalized"])
                #     # Show table for comparing
                # st.write(normalized_fig.to_html(escape = False), unsafe_allow_html = True)
            else:
                normalized = curate.normalize_groups(df)
                #redundante?
            #----------------------------------------------------------------------------------#
            # Neutralize when possible
            if options[1] in selected_options:

                st.header('**Neutralized Groups**')
                #if options[0] in selected_options:
                neutralized, original = curate.neutralize(normalized)
                cc.img_AgGrid(neutralized,name_smiles,curate.curated_smiles)

                # #Generate Image from normalized SMILES
                # PandasTools.AddMoleculeColumnToFrame(original, smilesCol = curate.curated_smiles,
                # molCol = "Normalized", includeFingerprints=False)
                # #Generate Image from Neutralized SMILES
                # PandasTools.AddMoleculeColumnToFrame(neutralized, smilesCol = curate.curated_smiles,
                # molCol = "Neutralized", includeFingerprints = False)
                # # Filter only columns containing images
                # neutralized_fig = neutralized.filter(items = ["Normalized", "Neutralized"])
                # # Show table for comparing
                # st.write(neutralized_fig.to_html(escape = False), unsafe_allow_html = True)
            else:
                neutralized,_ = curate.neutralize(normalized)

            #---------------------------------------------------------------------------------#
            # Remove mixtures and salts
            if options[2] in selected_options:

                st.header('**Remove mixtures**')
                # if options[1] in selected_options:
                no_mixture,only_mixture = curate.remove_mixture(neutralized)

                #Generate Image from Neutralized SMILES
                PandasTools.AddMoleculeColumnToFrame(only_mixture, smilesCol=curate.curated_smiles,
                molCol="Mixtures", includeFingerprints=False)
                #Generate Image from No_mixture SMILES
                # PandasTools.AddMoleculeColumnToFrame(no_mixture, smilesCol="no_mixture_smiles",
                # molCol="No_mixture", includeFingerprints=False)
                # Filter only columns containing images
                no_mixture_fig = only_mixture.filter(items=["Mixtures"])
                # Show table for comparing
                st.write(no_mixture_fig.to_html(escape=False), unsafe_allow_html=True)
            else:
                no_mixture,_ = curate.remove_mixture(neutralized)

            #---------------------------------------------------------------------------------#
            #Generate canonical tautomers
            if options[3] in selected_options:

                st.header('**Generate canonical tautomers**')
                # if options[2] in selected_options:
                canonical_tautomer,not_canon = curate.canonical_tautomer(no_mixture)

                # #Generate Image from Neutralized SMILES
                # PandasTools.AddMoleculeColumnToFrame(not_canon, smilesCol = curate.curated_smiles,
                # molCol = "No_mixture", includeFingerprints = False)
                # #Generate Image from No_mixture SMILES
                # PandasTools.AddMoleculeColumnToFrame(canonical_tautomer, smilesCol=curate.curated_smiles,
                # molCol = "Canonical_tautomer", includeFingerprints = False)
                # # Filter only columns containing images
                # canonical_tautomer_fig = canonical_tautomer.filter(items = ["No_mixture", "Canonical_tautomer"])
                # # Show table for comparing
                # st.write(canonical_tautomer_fig.to_html(escape = False), unsafe_allow_html = True)
            else:
                canonical_tautomer,_ = curate.canonical_tautomer(no_mixture)

            if options[4] in selected_options:
                st.header('**Standardized**')
                standardized,not_standardized = curate.standardise(canonical_tautomer)
                # #Generate Image from Neutralized SMILES
                # PandasTools.AddMoleculeColumnToFrame(standardized, smilesCol = curate.curated_smiles,
                # molCol="Standardized", includeFingerprints = False)
                # PandasTools.AddMoleculeColumnToFrame(not_standardized, smilesCol = curate.curated_smiles,
                # molCol="Not_Standardized", includeFingerprints = False)
                # #Generate Image from No_mixture SMILES
                # standardized_fig = standardized.filter(items = ["Standardized","Not_standardized"])
                # # Show table for comparing
                # st.write(standardized_fig.to_html(escape = False), unsafe_allow_html= True)
            else:
                standardized = curate.std_routine(canonical_tautomer)

            
        ########################################################################################################################################
        # Download Standardized with Duplicates
        ########################################################################################################################################
                
            # std_with_dup = canonical_tautomer.filter(items=["canonical_tautomer",])
            # std_with_dup.rename(columns={"canonical_tautomer": "SMILES",},inplace=True)
            # std_with_dup = std_with_dup.join(st.session_state.updated_df.drop(name_smiles, 1))

            filedownload(standardized,"Standardized with Duplicates")
            def duplicate_analysis(df,dups):
                st.header('**Duplicate Analysis**')
                st.write("Number of duplicates removed: ",dups)
                st.write("Number of compounds remaining: ",len(df))
                st.write("Percentage of compounds removed: ",round(dups/len(df)*100,2),"%")
                st.write("Percentage of compounds remaining: ",round(100-dups/len(df)*100,2),"%")
                st.header("**Final Dataset**")
                cc.AgGrid(df,key="final_dataset")
                filedownload(df,"Final Dataset")
            if radio == data_type[0]:
                continuous = utils.Continuous_Duplicate_Remover(standardized,curate.smiles,name_activity,False,False)
                continuous,dups = continuous.remove_duplicates()
                duplicate_analysis(continuous,dups)

            elif radio == data_type[1]:
                categorical = utils.Classification_Duplicate_Remover(standardized,curate.smiles,name_activity)
                categorical,dups = categorical.remove_duplicates()
                duplicate_analysis(categorical,dups)

    ########################################################################################################################################
    # Sidebar - Upload File and select columns
    ########################################################################################################################################
    # Upload File
    # df = custom.upload_file()
    st.write('---')

    #st.header('**Original input data**')

    # Read Uploaded file and convert to pandas
    if df is not None:
        file_dependent_code(df)
    else:
        pass
    ########################################################################################################################################
    # Analysis of duplicates
    ########################################################################################################################################
                
    #     # Generate InchiKey
    #     inchikey = smi_to_inchikey(canonical_tautomer)
    #     # concordance calculation
    #     no_dup_inchikey = len(inchikey.drop_duplicates(subset='inchikey', keep=False))
    #     discordance_dup = inchikey.groupby('inchikey').filter(lambda x: len(x[name_activity].unique())>1)
    #     num_dup = len(inchikey)-no_dup_inchikey

    #     if num_dup > 0:
    #         concordance = (num_dup-len(discordance_dup))/num_dup*100
    #     else:
    #         concordance = "no_duplicates"

    # #--------------------------- Removal of duplicates------------------------------#
         
    #     # Remove discordant duplicates, i.e., whre "outcome" is not unique
    #     no_dup = inchikey.groupby('inchikey').filter(lambda x: len(x[name_activity].unique()) <= 1)
    #     no_dup = no_dup.drop_duplicates(subset='inchikey', keep="first")

    # #--------------------------- Print analysis of duplicates------------------------------#

    #     # counting compounds
    #     num_cdps_input = int(len(df[name_smiles]))
    #     num_cdps_stand = int(len(inchikey))
    #     num_cdps_no_duplicates = int(len(no_dup))
    #     num_of_duplicates = int(num_dup)
    #     discordant_dup = int(len(discordance_dup))
          
    #     # Dataframe counting compounds
    #     duplicate_analysis = pd.DataFrame([[num_cdps_input, num_cdps_stand, num_cdps_no_duplicates, num_of_duplicates, discordant_dup, concordance]], 
    #     columns=['num_cdps_input', 'num_cdps_stand', 'num_cdps_no_duplicates', 'num_of_duplicates','discordant_dup', 'Concordance (%)'])

    #     # Print analysis of duplicates
    #     st.header('**Analysis of Duplicates**')
            
    #     AgGrid(duplicate_analysis)
    #     filedownload(duplicate_analysis,"Duplicate analysis")
    # #--------------------------- Print dataframe without duplicates------------------------------#
    #     #Keep only curated smiles and outcome
    #     no_dup = no_dup.filter(items=["canonical_tautomer", name_activity])
    #     no_dup.rename(columns={"canonical_tautomer": "SMILES",},inplace=True)
    #     no_dup = no_dup.join(st.session_state.updated_df.drop([name_smiles, name_activity], 1))

    #     st.header('**Duplicates removed**')
    #     # Display curated dataset
    #     AgGrid(no_dup)

    # ########################################################################################################################################
    # # Data download
    # ########################################################################################################################################

    #     # File download
    #     filedownload(no_dup,"Curated")
