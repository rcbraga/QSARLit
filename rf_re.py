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
import warnings
warnings.filterwarnings(action='ignore')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

import numpy as np
from numpy import sqrt
from numpy import argmax

import pandas as pd

import matplotlib.pyplot as plt

import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import KFold, cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, roc_curve, roc_auc_score, make_scorer
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
import pickle
from sklearn.calibration import calibration_curve

from imblearn.metrics import geometric_mean_score

import multiprocessing

from skopt import BayesSearchCV
import utils
import plotly.graph_objects as go

def app(df, s_state):
    ########################################################################################################################################
    # Functions
    ########################################################################################################################################
    def getNeighborsDitance(trainingSet, testInstance, k):
        neighbors_k=metrics.pairwise.pairwise_distances(trainingSet, Y=testInstance, metric='dice', n_jobs=1)
        neighbors_k.sort(0)
        similarity= 1-neighbors_k
        return similarity[k-1,:]


    ########################################################################################################################################
    # Seed
    ########################################################################################################################################

    # Choose the general hyperparameters interval to be tested

    with st.sidebar.header('1. Set seed for reprodutivity'):
        parameter_random_state = st.sidebar.number_input('Seed number (random_state)', min_value=None, max_value=None, value=int(42))


    ########################################################################################################################################
    # Sidebar - Upload File and select columns
    ########################################################################################################################################

    # Upload File
    # with st.sidebar.header('2. Upload your CSV data (calculated descriptors)'):
    #     uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    # st.sidebar.markdown("""
    # [Example CSV input file](https://github.com/joseteofilo/data_qsarlit/blob/master/descriptor_morgan_r2_2048bits_for_modeling.csv)
    # """)
    cc = utils.Custom_Components()
    #df = cc.upload_file(key="input_file")

    # Read Uploaded file and convert to pandas
    if df is not None:
        # Read CSV data
        #df = pd.read_csv(uploaded_file, sep=',')

        #st.header('**Molecular descriptors input data**')

        #cc.AgGrid(df)

    # else:
    #     st.info('Awaiting for CSV file to be uploaded.')

        st.sidebar.write('---')

        # Select columns
        with st.sidebar.header('2. Enter column name in modeling set'):
            name_activity = st.sidebar.text_input('Enter column with activity (e.g., Active and Inactive that should be 1 and 0, respectively)', 'Outcome')

        st.sidebar.write('---')

        ########################################################################################################################################
        # Data splitting
        ########################################################################################################################################
        with st.sidebar.header('3. Select data splitting'):
            # Select fingerprint
            splitting_dict = {'Only k-fold':'kfold',
                    'k-fold and external set':'split_original',
                    'Input your own external set':'input_own',}
            user_splitting  = st.sidebar.selectbox('Choose an splitting', list(splitting_dict.keys()))
            selected_splitting = splitting_dict[user_splitting]
        
        with st.sidebar.subheader('3.1 Number of folds'):
            n_plits = st.sidebar.number_input('Enter the number of folds', min_value=None, max_value=None, value=int(5))

        # Selecting x and y from input file
        if df is not None:

            if selected_splitting == 'kfold':
                x = df.iloc[:, df.columns != name_activity].values  # Using all column except for the last column as X
                y = df[name_activity].values  # Selecting the last column as Y

            if selected_splitting == 'split_original':
                x = df.iloc[:, df.columns != name_activity].values  # Using all column except for the last column as X
                y = df[name_activity].values  # Selecting the last column as Y

                with st.sidebar.header('Test size (%)'):
                    input_test_size = st.sidebar.number_input('Enter the test size (%)', min_value=None, max_value=None, value=(20))
                    test_size = input_test_size/100
                    x, x_ext, y, y_ext = train_test_split(x, y, test_size=test_size, random_state=0, stratify=y)

            if selected_splitting == 'input_own':

                # Upload File
                own_external = cc.upload_file(custom_title="3.2 Upload your CSV of external set (calculated descriptors)", key='own_external',context=st.sidebar)
                    # Read Uploaded file and convert to pandas
                if own_external is not None:
                    #st.button('Show external set?')
                    if st.button('Show external set?', key='button_own_external',on_click=lambda: False if s_state.button_own_external else True):
                        st.header('**External set**')
                        cc.AgGrid(own_external)
                    with st.sidebar.header('3.3 Enter column name'):
                        name_activity_ext = st.sidebar.text_input('Enter column with activity in externl set (e.g., Active and Inactive that should be 1 and 0, respectively)', 'Outcome')

                    st.sidebar.write('---')

                    # Read CSV data
                    df_own = own_external

                    # st.header('**Molecular descriptors of external set**')
                    # st.write(df_own)

                    x = df.iloc[:, df.columns != name_activity].values  # Using all column except for the last column as X
                    y = df[name_activity].values  # Selecting the last column as Y

                    x_ext = df_own.iloc[:, df_own.columns != name_activity_ext].values  # Using all column except for the last column as X
                    y_ext = df_own[name_activity_ext].values  # Selecting the last column as Y


        ########################################################################################################################################
        # Sidebar - Specify parameter settings
        ########################################################################################################################################

        st.sidebar.header('5. Set Parameters - Bayesian hyperparameter search')

        # Choose the general hyperparameters
        st.sidebar.subheader('General Parameters')

        parameter_n_iter = st.sidebar.slider('Number of iterations (n_iter)', 1, 1000, 3, 1)
        st.sidebar.write('---')
        parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[-1, 1])

        # Select the hyperparameters to be optimized
        st.sidebar.subheader('Select the hyperparameters to be optimized')

        container = st.sidebar.container()
        slc_all = st.sidebar.checkbox("Select all")
        #cleaning some code
        rf_hyperparams=['n_estimators','max_depth','min_samples_split','min_samples_leaf']

        if slc_all:
            selected_options = container.multiselect("Select one or more options:", rf_hyperparams, rf_hyperparams)
        else:
            selected_options =  container.multiselect("Select one or more options:", rf_hyperparams)

        #st.write(selected_options)

        st.sidebar.write('---')

        # Choose the hyperparameters intervals to be tested
        st.sidebar.subheader('Learning Hyperparameters')

        if selected_options is None:
            st.sidebar.write('Select hyperparameters')

        selected_hyperparameters = {}

        if rf_hyperparams[0] in selected_options:
            n_estimator_container=st.sidebar.container()
            try:
                min_parameter_n_estimators = n_estimator_container.number_input('Minimal value of estimators (n_estimators)', 50, max_value=None, step=1)
                max_parameter_n_estimators = n_estimator_container.number_input('Maximum value of estimators (n_estimators)', min_parameter_n_estimators, max_value=None, step=1)
            except:
                n_estimator_container.write("First value (minimum) must be smaller than second(maximum) value")
            n_estimators = {'n_estimators': [min_parameter_n_estimators, max_parameter_n_estimators]}
            selected_hyperparameters.update(n_estimators)
            #st.write(selected_hyperparameters)
            st.sidebar.write('---')

        if rf_hyperparams[1] in selected_options:
            min_parameter_max_depth = st.sidebar.number_input('Minimum Max depth (max_depth)', 1, 200)
            max_parameter_max_depth = st.sidebar.number_input('Maximum Max depth (max_depth)', min_parameter_max_depth, 200)
            max_depth = {"max_depth": [min_parameter_max_depth, max_parameter_max_depth]}
            selected_hyperparameters.update(max_depth)
            #st.write(selected_hyperparameters)
            st.sidebar.write('---')

        if rf_hyperparams[2] in selected_options:
            min_parameter_min_samples_split = st.sidebar.number_input('Minimum number of samples required to split an internal node (min_samples_split)', 2, 50)
            max_parameter_min_samples_split = st.sidebar.number_input('Maximum number of samples required to split an internal node (min_samples_split)', min_parameter_min_samples_split, 50)
            min_samples_split = {'min_samples_split': [min_parameter_min_samples_split, max_parameter_min_samples_split]}
            selected_hyperparameters.update(min_samples_split)
            #st.write(selected_hyperparameters)
            st.sidebar.write('---')

        if rf_hyperparams[3] in selected_options:
            min_parameter_min_samples_leaf = st.sidebar.number_input('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 50)
            max_parameter_min_samples_leaf = st.sidebar.number_input('Maximum number of samples required to be at a leaf node (min_samples_leaf)', min_parameter_min_samples_leaf, 50)
            min_samples_leaf = {'min_samples_leaf': [min_parameter_min_samples_leaf,max_parameter_min_samples_leaf]}
            selected_hyperparameters.update(min_samples_leaf)
            #st.write(selected_hyperparameters)
            st.sidebar.write('---')

        else:
            st.sidebar.write('Please, select the hyperparameters to be optimized!')

        ########################################################################################################################################
        # Modeling
        ########################################################################################################################################

        if st.sidebar.button('Run Modeling'):

        #---------------------------------#
        #Create folds for cross-validation
            cv = KFold(n_splits = n_plits, shuffle = False)

        #---------------------------------
        # Run RF Model building - Bayesian hyperparameter search
            scorer = make_scorer(geometric_mean_score)

            # log-uniform: understand as search over p = exp(x) by varying x
            opt_rf = BayesSearchCV(
                RandomForestRegressor(),
                selected_hyperparameters,
                n_iter=parameter_n_iter, # Number of parameter settings that are sampled
                cv=cv,
                scoring = "neg_root_mean_squared_error",
                verbose=0,
                refit= True, # Refit the best estimator with the entire dataset.
                random_state=parameter_random_state,
                n_jobs = parameter_n_jobs
            )

            opt_rf.fit(x, y)

            st.write("Best parameters: %s" % opt_rf.best_params_)

        #---------------------------------#
        # k-fold cross-validation

            scoring = ['max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
            scores = cross_validate(opt_rf, x, y, cv=5, scoring=scoring)
        #---------------------------------#
        # Statistics k-fold cross-validation

            statistics = {}

            statistics['MAX ERROR'] = round(scores['test_max_error'].mean()*-1, 2) 
            statistics['MAE'] = round(scores['test_neg_mean_absolute_error'].mean()*-1, 2) 
            statistics['MSE'] = round(scores['test_neg_mean_squared_error'].mean()*-1, 2) 
            statistics['RMSE'] = round(scores['test_neg_root_mean_squared_error'].mean()*-1, 2) 
            statistics['R2'] = round(scores['test_r2'].mean(), 2)

        #---------------------------------#
        #converting calculated metrics into a pandas dataframe to save a xls
            model_type = "RF"

            result_type = "uncalibrated"

            metrics_rf_uncalibrated = statistics
            metrics_rf_uncalibrated['model'] = model_type
            metrics_rf_uncalibrated['result_type'] = result_type

            st.header('**Metrics of uncalibrated model on the K-fold cross validation**')
            
            #---------------------------------#
            # Bar chart Statistics k-fold cross-validation
            metrics_rf_uncalibrated = pd.DataFrame([metrics_rf_uncalibrated], columns=metrics_rf_uncalibrated.keys())

            metrics_rf_uncalibrated_graph = metrics_rf_uncalibrated.filter(items=['MAX ERROR', 'MAE', 'MSE', 'RMSE', 'R2'])
            
            x = metrics_rf_uncalibrated_graph.columns
            y = metrics_rf_uncalibrated_graph.loc[0].values
            
            colors = ["red", "orange", "green", 'yellow', "pink", 'blue', "purple", "cyan", "teal"]

            fig = go.Figure(data=[go.Bar(
                x=x, y=y,
                text=y,
                textposition='auto',
                marker_color = colors
            )])

            st.plotly_chart(fig)   
            
            ########################################################################################################################################
            # External set uncalibrated
            ########################################################################################################################################
            if selected_splitting == 'split_original' or selected_splitting == 'input_own':
                            
                # Predict probabilities for the external set
                probs_external = opt_rf.predict(x_ext)
                # Making classes
                pred_rf = (probs_external[:, 1] > 0.5).astype(int)
                # Statistics external set uncalibrated
                scores = cross_validate(opt_rf, pred_rf, y_ext, cv=5, scoring=scoring)

                statistics = {}

                statistics['MAX ERROR'] = round(scores['test_max_error'].mean()*-1, 2) 
                statistics['MAE'] = round(scores['test_neg_mean_absolute_error'].mean()*-1, 2) 
                statistics['MSE'] = round(scores['test_neg_mean_squared_error'].mean()*-1, 2) 
                statistics['RMSE'] = round(scores['test_neg_root_mean_squared_error'].mean()*-1, 2) 
                statistics['R2'] = round(scores['test_r2'].mean(), 2)
                
                #---------------------------------#
                #converting calculated metrics into a pandas dataframe to save a xls
                model_type = "RF"
                
                result_type = "uncalibrated_external_set"

                metrics_rf_external_set_uncalibrated = statistics
                metrics_rf_external_set_uncalibrated['model'] = model_type
                metrics_rf_external_set_uncalibrated['result_type'] = result_type
                

                st.header('**Metrics of uncalibrated model on the external set**') 
                #---------------------------------#
                # Bar chart Statistics k-fold cross-validation
                metrics_rf_external_set_uncalibrated = pd.DataFrame([metrics_rf_uncalibrated], columns=metrics_rf_uncalibrated.keys())
                metrics_rf_external_set_uncalibrated_graph = metrics_rf_external_set_uncalibrated.filter(items=['MAX ERROR', 'MAE', 'MSE', 'RMSE', 'R2'])
                
                x = metrics_rf_external_set_uncalibrated_graph.columns
                y = metrics_rf_external_set_uncalibrated_graph.loc[0].values
                
                colors = ["red", "orange", "green", 'yellow', "pink", 'blue', "purple", "cyan", "teal"]

                fig = go.Figure(data=[go.Bar(
                    x=x, y=y,
                    text=y,
                    textposition='auto',
                    marker_color = colors
                )])

                st.plotly_chart(fig)                          


            ########################################################################################################################################
            # Model Calibration
            ########################################################################################################################################        
            #---------------------------------#

            ########################################################################################################################################
            # External set calibrated
            ########################################################################################################################################

        ########################################################################################################################################
        # Compare models
        ########################################################################################################################################

        ########################################################################################################################################
        # Download files
        ########################################################################################################################################

            st.header('**Download files**')
                    
            if selected_splitting == 'split_original' or selected_splitting == 'input_own':
                frames = [metrics_rf_uncalibrated, metrics_rf_external_set_uncalibrated]

            else:
                frames = [metrics_rf_uncalibrated]

        
            result = pd.concat(frames)

            result = result.round(2)

            # File download
            def filedownload(df):
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
                href = f'<a href="data:file/csv;base64,{b64}" download="metrics_rf.csv">Download CSV File - metrics</a>'
                st.markdown(href, unsafe_allow_html=True)

            filedownload(result)

            def download_model(model):
                output_model = pickle.dumps(model)
                b64 = base64.b64encode(output_model).decode()
                href = f'<a href="data:file/output_model;base64,{b64}" download= model_rf.pkl >Download generated model (PKL File)</a>'
                st.markdown(href, unsafe_allow_html=True)

            download_model(opt_rf)