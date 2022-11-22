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

#from rdkit import Chem, DataStructs

import numpy as np
from numpy import sqrt
from numpy import argmax

import pandas as pd

import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn.model_selection import train_test_split, StratifiedKFold
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

def app(df,s_state):
    ########################################################################################################################################
    # Functions
    ########################################################################################################################################
    def getNeighborsDitance(trainingSet, testInstance, k):
        neighbors_k=metrics.pairwise.pairwise_distances(trainingSet, Y=testInstance, metric='dice', n_jobs=1)
        neighbors_k.sort(0)
        similarity= 1-neighbors_k
        return similarity[k-1,:]

    #5-fold-cross-val
    def cros_val(x,y,classifier):
        probs_classes = []
        y_test_all = []
        AD_fold =[]
        distance_train_set =[]
        distance_test_set = []
        y_pred_ad=[]
        y_exp_ad =[]
        for train_index, test_index in cv.split(x, y):
            clf = classifier # model with best parameters
            X_train_folds = x[train_index] # descritors train split
            y_train_folds = np.array(y)[train_index.astype(int)] # label train split
            X_test_fold = x[test_index] # descritors test split
            y_test_fold = np.array(y)[test_index.astype(int)] # label test split
            clf.fit(X_train_folds, y_train_folds) # train fold
            y_pred = clf.predict_proba(X_test_fold) # test fold
            probs_classes.append(y_pred) # all predictions for test folds
            y_test_all.append(y_test_fold) # all folds' labels
            k= int(round(pow((len(y)) ,1.0/3), 0))
            distance_train = getNeighborsDitance(X_train_folds, X_train_folds, k)
            distance_train_set.append(distance_train)
            distance_test = getNeighborsDitance(X_train_folds, X_test_fold, k)
            distance_test_set.append(distance_test)
            Dc = np.average(distance_train)-(0.5*np.std(distance_train))
            for i in range(len(X_test_fold)):
                ad=0
                if distance_test_set[0][i] >= Dc:
                    ad = 1
                AD_fold.append(ad)
        probs_classes = np.concatenate(probs_classes)
        y_experimental = np.concatenate(y_test_all)
        # Uncalibrated model predictions
        pred = (probs_classes[:, 1] > 0.5).astype(int)
        for i in range(len(AD_fold)):
            if AD_fold[i] == 1:
                y_pred_ad.append(pred[i])
                y_exp_ad.append(y_experimental[i])

        return(pred, y_experimental, probs_classes, AD_fold, y_pred_ad, y_exp_ad)

    #STATISTICS
    def calc_statistics(y,pred):
        # save confusion matrix and slice into four pieces
        confusion = confusion_matrix(y, pred)
        #[row, column]
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]

        # calc statistics
        accuracy = round(accuracy_score(y, pred),2)#accuracy
        mcc = round(matthews_corrcoef(y, pred),2) #mcc
        kappa = round(cohen_kappa_score(y, pred),2) #kappa
        sensitivity = round(recall_score(y, pred),2) #Sensitivity
        specificity = round(TN / (TN + FP),2) #Specificity
        positive_pred_value = round(TP / float(TP + FP),2) #PPV
        negative_pred_value = round(TN / float(TN + FN),2) #NPV
        auc = round(roc_auc_score(y, pred),2) #AUC
        bacc = round(balanced_accuracy_score(y, pred),2) # balanced accuracy

        #converting calculated metrics into a pandas dataframe to compare all models at the final
        statistics = pd.DataFrame({'Bal-acc': bacc, "Sensitivity": sensitivity, "Specificity": specificity,"PPV": positive_pred_value,
               "NPV": negative_pred_value, 'Kappa': kappa, 'AUC': auc, 'MCC': mcc, 'Accuracy': accuracy,}, index=[0])
        return(statistics)

    cc = utils.Custom_Components()
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
    #with st.sidebar.header('2. Upload your CSV data (calculated descriptors)'):
    uploaded_file = cc.upload_file(custom_title="2. Upload your CSV data (calculated descriptors)",context=st.sidebar, key="descriptor",file_type=["csv"])
    

    # Read Uploaded file and convert to pandas
    if uploaded_file is not None:
        # Read CSV data
        df = uploaded_file

        #st.header('**Molecular descriptors input data**')

        #cc.AgGrid(df)

    
    st.sidebar.write('---')

    # Select columns
    with st.sidebar.header('3. Enter column name in modeling set'):
        name_activity = st.sidebar.text_input('Enter column with activity (e.g., Active and Inactive that should be 1 and 0, respectively)', 'Outcome')

    st.sidebar.write('---')

    ########################################################################################################################################
    # Data splitting
    ########################################################################################################################################
    with st.sidebar.header('4. Select data splitting'):
        # Select fingerprint
        splitting_dict = {'Only k-fold':'kfold',
                   'k-fold and external set':'split_original',
                   'Input your own external set':'input_own',}
        user_splitting  = st.sidebar.selectbox('Choose an splitting', list(splitting_dict.keys()))
        selected_splitting = splitting_dict[user_splitting]
    
    with st.sidebar.subheader('4.1 Number of folds'):
        n_plits = st.sidebar.number_input('Enter the number of folds', min_value=None, max_value=None, value=int(5))

    # Selecting x and y from input file
    if uploaded_file is not None:

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
            own_external = cc.upload_file(custom_title="3.2 Upload your CSV of external set (calculated descriptors)",key="own_external",context=st.sidebar, type=["csv"])

                # Read Uploaded file and convert to pandas
            if own_external is not None:
            
                with st.sidebar.header('4.3 Enter column name'):
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
    lgbm_hyperparams=['max_depth', 'max_bin', 'num_leaves', 'learning_rate', 'n_estimators', 
    'feature_fraction', 'min_child_weight', 'min_child_samples', 'colsample_bytree']

    if slc_all:
        selected_options = container.multiselect("Select one or more options:", lgbm_hyperparams, lgbm_hyperparams)
    else:
        selected_options =  container.multiselect("Select one or more options:", lgbm_hyperparams)

    #st.write(selected_options)

    st.sidebar.write('---')

    # Choose the hyperparameters intervals to be tested
    st.sidebar.subheader('Learning Hyperparameters')

    if selected_options is None:
        st.sidebar.write('Select hyperparameters')

    selected_hyperparameters = {}

    if lgbm_hyperparams[0] in selected_options:

        min_parameter_max_depth = st.sidebar.number_input('Minimum value of Max depth (max_depth)', 1, 200)
        max_parameter_max_depth = st.sidebar.number_input('Maximum value of Max depth (max_depth)', 30, 200)
        max_depth = {"max_depth": [min_parameter_max_depth, max_parameter_max_depth]}
        selected_hyperparameters.update(max_depth)
        st.sidebar.write('---')

    if lgbm_hyperparams[1] in selected_options:
        min_parameter_max_bin = st.sidebar.number_input('Minimum value of Max_bin', 1, 500)
        max_parameter_max_bin = st.sidebar.number_input('Maximum value of Max_bin', 500, 500)
        max_bin = {"max_bin": [min_parameter_max_bin, max_parameter_max_bin]}
        selected_hyperparameters.update(max_bin)
        st.sidebar.write('---')

    if lgbm_hyperparams[2] in selected_options:
        min_parameter_num_leaves = st.sidebar.number_input('Minimum number of decision leaves (num_leaves)', 31, 80)
        max_parameter_num_leaves = st.sidebar.number_input('Maximum number of decision leaves (num_leaves)', 80, 80)
        num_leaves = {'num_leaves': [min_parameter_num_leaves, max_parameter_num_leaves]}
        selected_hyperparameters.update(num_leaves)
        st.sidebar.write('---')

    if lgbm_hyperparams[3] in selected_options:
        min_parameter_learning_rate = st.sidebar.number_input('Minimum number of learning rate (learning_rate)', 0.001, 0.35)
        max_parameter_learning_rate = st.sidebar.number_input('Maximum number of learning rate (learning_rate)', 0.35, 0.35)
        learning_rate = {'learning_rate': [min_parameter_learning_rate,max_parameter_learning_rate]}
        selected_hyperparameters.update(learning_rate)
        st.sidebar.write('---')

    if lgbm_hyperparams[4] in selected_options:

        n_estimator_container=st.sidebar.container()
        try:
            min_parameter_n_estimators = n_estimator_container.number_input('Minimal value of estimators (n_estimators)', 50, max_value=None, step=1)
            max_parameter_n_estimators = n_estimator_container.number_input('Maximum value of estimators (n_estimators)', 500, max_value=None, step=1)
        except:
            n_estimator_container.write("First value (minimum) must be smaller than second(maximum) value")
        n_estimators = {'n_estimators': [min_parameter_n_estimators, max_parameter_n_estimators]}
        selected_hyperparameters.update(n_estimators)
        st.sidebar.write('---')

    if lgbm_hyperparams[5] in selected_options:
        min_parameter_feature_fraction = st.sidebar.number_input('Minimum number of a subset of features on each iteration (feature_fraction)', 0.7, 0.999)
        max_parameter_feature_fraction = st.sidebar.number_input('Maximum number of a subset of features on each iteration (feature_fraction)', 0.999, 0.999)
        feature_fraction = {'feature_fraction': [min_parameter_feature_fraction,max_parameter_feature_fraction]}
        selected_hyperparameters.update(feature_fraction)
        st.sidebar.write('---')

    if lgbm_hyperparams[6] in selected_options:
        min_parameter_min_child_weight = st.sidebar.number_input('Minimum value for minimum sum of instance weight (hessian) needed in a child (leaf)', 1, 50)
        max_parameter_min_child_weight = st.sidebar.number_input('Maximum value for minimum sum of instance weight (hessian) needed in a child (leaf) (min_child_weight)', 10, 50)
        min_child_weight = {'min_child_weight': [min_parameter_min_child_weight,max_parameter_min_child_weight]}
        selected_hyperparameters.update(min_child_weight)
        st.sidebar.write('---')

    if lgbm_hyperparams[7] in selected_options:
        min_parameter_min_child_samples = st.sidebar.number_input('Minimum value for minimum number of data needed in a child (leaf) - min_child_samples', 1, 50)
        max_parameter_min_child_samples = st.sidebar.number_input('Maximum value for minimum number of data needed in a child (leaf) - min_child_samples', 20, 50)
        min_child_samples = {'min_child_samples': [min_parameter_min_child_samples,max_parameter_min_child_samples]}
        selected_hyperparameters.update(min_child_samples)
        st.sidebar.write('---')

    if lgbm_hyperparams[8] in selected_options:
        min_parameter_colsample_bytree = st.sidebar.number_input('Minimum value percentage of features before training each tree - colsample_bytree', 0.7, 1.0)
        max_parameter_colsample_bytree = st.sidebar.number_input('Maximum value fpercentage of features before training each tree - colsample_bytree', 1.0, 1.0)
        colsample_bytree = {'colsample_bytree': [min_parameter_colsample_bytree,max_parameter_colsample_bytree]}
        selected_hyperparameters.update(colsample_bytree)
        st.sidebar.write('---')

    else:
        st.sidebar.write('Please, select the hyperparameters to be optimized!')

    ########################################################################################################################################
    # Modeling
    ########################################################################################################################################

    if st.sidebar.button('Run Modeling'):

    #---------------------------------#
    #Create folds for cross-validation
        cv = StratifiedKFold(n_splits = n_plits, shuffle=False,)

    #---------------------------------#
    # Run RF Model building - Bayesian hyperparameter search
        scorer = make_scorer(geometric_mean_score)

        # log-uniform: understand as search over p = exp(x) by varying x
        opt_lgbm = BayesSearchCV(
            lgb.LGBMClassifier(),
            selected_hyperparameters,
            n_iter=parameter_n_iter, # Number of parameter settings that are sampled
            cv=cv,
            scoring = scorer,
            verbose=0,
            refit= True, # Refit the best estimator with the entire dataset.
            random_state=parameter_random_state,
            n_jobs = parameter_n_jobs
        )

        opt_lgbm.fit(x, y)

        st.write("Best parameters: %s" % opt_lgbm.best_params_)

    #---------------------------------#
    # k-fold cross-validation
        pred_lgbm, y_experimental, probs_classes, AD_fold, y_pred_ad, y_exp_ad = cros_val(x,y, lgb.LGBMClassifier(**opt_lgbm.best_params_))
    #---------------------------------#
    # Statistics k-fold cross-validation
        statistics = calc_statistics(y_experimental, pred_lgbm)
    #---------------------------------#
    # coverage
        coverage = round((len(y_exp_ad)/len(y_experimental)),2)

    #---------------------------------#
    #converting calculated metrics into a pandas dataframe to save a xls
        model_type = "LGBM"

        result_type = "uncalibrated"

        metrics_lgbm_uncalibrated = statistics
        metrics_lgbm_uncalibrated['model'] = model_type
        metrics_lgbm_uncalibrated['result_type'] = result_type
        metrics_lgbm_uncalibrated['coverage'] = coverage

        st.header('**Metrics of uncalibrated model on the K-fold cross validation**')
        
        #---------------------------------#
        # Bar chart Statistics k-fold cross-validation

        metrics_lgbm_uncalibrated_graph = metrics_lgbm_uncalibrated.filter(items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC", "coverage"])
        
        x = metrics_lgbm_uncalibrated_graph.columns
        y = metrics_lgbm_uncalibrated_graph.loc[0].values
        
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
            probs_external = opt_lgbm.predict_proba(x_ext)
            # Making classes
            pred_lgbm = (probs_external[:, 1] > 0.5).astype(int)
            # Statistics external set uncalibrated
            statistics = calc_statistics(y_ext, pred_lgbm)
            
            #---------------------------------#
            #converting calculated metrics into a pandas dataframe to save a xls
            model_type = "LGBM"
            
            result_type = "uncalibrated_external_set"

            metrics_lgbm_external_set_uncalibrated = statistics
            metrics_lgbm_external_set_uncalibrated['model'] = model_type
            metrics_lgbm_external_set_uncalibrated['result_type'] = result_type
            

            st.header('**Metrics of uncalibrated model on the external set**') 
            #---------------------------------#
            # Bar chart Statistics k-fold cross-validation

            metrics_lgbm_external_set_uncalibrated_graph = metrics_lgbm_external_set_uncalibrated.filter(items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC", "coverage"])
            
            x = metrics_lgbm_external_set_uncalibrated_graph.columns
            y = metrics_lgbm_external_set_uncalibrated_graph.loc[0].values
            
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
        # Check model calibatrion
        # keep probabilities for the positive outcome only
        probs = probs_classes[:, 1]
        # reliability diagram
        fop, mpv = calibration_curve(y_experimental, probs, n_bins=10)
        # plot perfectly calibrated
        fig = plt.figure()
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot model reliability
        plt.plot(mpv, fop, marker='.')
        
        st.header('**Check model calibatrion**')
        st.pyplot(fig)
        
        #---------------------------------#
        # Use ROC-Curve and Gmean to select a threshold for calibration
        # keep probabilities for the positive outcome only
        yhat = probs_classes[:, 1]
        # calculate roc curves
        fpr, tpr, thresholds = roc_curve(y_experimental, yhat)
        # calculate the g-mean for each threshold
        gmeans = sqrt(tpr * (1-fpr))
        # locate the index of the largest g-mean
        ix = argmax(gmeans)
        # plot the roc curve for the model
        fig = plt.figure()
        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, marker='.', label='RF')
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()

        st.header('**Use ROC-Curve and Gmean to select a threshold for calibration**')

        st.pyplot(fig)
     
        st.write('Best Threshold= %.2f, G-Mean= %.2f' % (round(thresholds[ix], 2), round(gmeans[ix], 2)))
        
        #---------------------------------#
        # Record the threshold in a variable
        threshold_roc = round(thresholds[ix], 2)

        #---------------------------------#
        # Select the best threshold to distinguishthe classes
        pred_lgbm = (probs_classes[:, 1] > threshold_roc).astype(int)
        
        #---------------------------------#
        # Statistics Statistics k-fold cross-validation calibrated
        statistics = calc_statistics(y_experimental, pred_lgbm)
        
        #---------------------------------#
        # Coverage
        coverage = round((len(y_exp_ad)/len(y_experimental)),2)

        #---------------------------------#
        #converting calculated metrics into a pandas dataframe to save a xls
        model_type = "LGBM"

        result_type = "calibrated"

        metrics_lgbm_calibrated = statistics
        metrics_lgbm_calibrated['model'] = model_type
        metrics_lgbm_calibrated['result_type'] = result_type
        metrics_lgbm_calibrated['calibration_threshold'] = threshold_roc
        metrics_lgbm_calibrated['coverage'] = coverage

        st.header('**Metrics of calibrated model on the K-fold cross validation**')

        #---------------------------------#
        # Bar chart Statistics k-fold cross-validation calibrated
    
        metrics_lgbm_calibrated_graph = metrics_lgbm_calibrated.filter(items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC", "coverage"])
        x = metrics_lgbm_calibrated_graph.columns
        y = metrics_lgbm_calibrated_graph.loc[0].values
       
        colors = ["red", "orange", "green", 'yellow', "pink", 'blue', "purple", "cyan", "teal"]

        fig = go.Figure(data=[go.Bar(
            x=x, y=y,
            text=y,
            textposition='auto',
            marker_color = colors
        )])

        st.plotly_chart(fig)

        ########################################################################################################################################
        # External set calibrated
        ########################################################################################################################################
        if selected_splitting == 'split_original' or selected_splitting == 'input_own':
                        
            # Predict probabilities for the external set
            probs_external = opt_lgbm.predict_proba(x_ext)
            # Making classes
            pred_lgbm = (probs_external[:, 1] > threshold_roc).astype(int)
            # Statistics external set uncalibrated
            statistics = calc_statistics(y_ext, pred_lgbm)
            
            #---------------------------------#
            #converting calculated metrics into a pandas dataframe to save a xls
            model_type = "LGBM"
            
            result_type = "calibrated_external_set"

            metrics_lgbm_external_set_calibrated = statistics
            metrics_lgbm_external_set_calibrated['model'] = model_type
            metrics_lgbm_external_set_calibrated['result_type'] = result_type
            

            st.header('**Metrics of calibrated model on the external set**') 
            #---------------------------------#
            # Bar chart Statistics k-fold cross-validation

            metrics_lgbm_external_set_calibrated_graph = metrics_lgbm_external_set_calibrated.filter(items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC", "coverage"])
            
            x = metrics_lgbm_external_set_calibrated_graph.columns
            y = metrics_lgbm_external_set_calibrated_graph.loc[0].values
            
            colors = ["red", "orange", "green", 'yellow', "pink", 'blue', "purple", "cyan", "teal"]

            fig = go.Figure(data=[go.Bar(
                x=x, y=y,
                text=y,
                textposition='auto',
                marker_color = colors
            )])

            st.plotly_chart(fig)                          

    ########################################################################################################################################
    # Compare models
    ########################################################################################################################################

        # Only K-fold
        st.header('**Compare metrics of calibrated and uncalibrated models on the k-fold cross validation**')

        metrics_lgbm_uncalibrated_graph = metrics_lgbm_uncalibrated.filter(items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC"])
        metrics_lgbm_calibrated_graph = metrics_lgbm_calibrated.filter(items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC"])

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
              r=metrics_lgbm_uncalibrated_graph.loc[0].values,
              theta=metrics_lgbm_uncalibrated_graph.columns,
              fill='toself',
              name='Uncalibrated'
        ))
        fig.add_trace(go.Scatterpolar(
              r=metrics_lgbm_calibrated_graph.loc[0].values,
              theta=metrics_lgbm_uncalibrated_graph.columns,
              fill='toself',
              name='Calibrated'
        ))

        fig.update_layout(
          polar=dict(
            radialaxis=dict(
              visible=True,
              range=[0, 1]
            )),
          showlegend=True
        )

        st.plotly_chart(fig)

        #---------------------------------#
        # External set

        if selected_splitting == 'split_original' or selected_splitting == 'input_own':
            
            st.header('**Compare metrics of calibrated and uncalibrated models on the external set**')

            metrics_lgbm_external_set_uncalibrated_graph = metrics_lgbm_external_set_uncalibrated.filter(items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC"])
            metrics_lgbm_external_set_calibrated_graph = metrics_lgbm_external_set_calibrated.filter(items=['Bal-acc', "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "MCC", "AUC"])

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=metrics_lgbm_external_set_uncalibrated_graph.loc[0].values,
                theta=metrics_lgbm_external_set_uncalibrated_graph.columns,
                fill='toself',
                name='Uncalibrated'
            ))
            fig.add_trace(go.Scatterpolar(
                r=metrics_lgbm_external_set_calibrated_graph.loc[0].values,
                theta=metrics_lgbm_external_set_calibrated_graph.columns,
                fill='toself',
                name='Calibrated'
            ))

            fig.update_layout(
            polar=dict(
                radialaxis=dict(
                visible=True,
                range=[0, 1]
                )),
            showlegend=True
            )

            st.plotly_chart(fig)

    ########################################################################################################################################
    # Download files
    ########################################################################################################################################

        st.header('**Download files**')
                
        if selected_splitting == 'split_original' or selected_splitting == 'input_own':
            frames = [metrics_lgbm_uncalibrated, metrics_lgbm_calibrated, 
            metrics_lgbm_external_set_uncalibrated, metrics_lgbm_external_set_calibrated]

        else:
            frames = [metrics_lgbm_uncalibrated, metrics_lgbm_calibrated,]

       
        result = pd.concat(frames)

        result = result.round(2)

        # File download
        def filedownload(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download="metrics_lgbm.csv">Download CSV File - metrics</a>'
            st.markdown(href, unsafe_allow_html=True)

        filedownload(result)

        def download_model(model):
            output_model = pickle.dumps(model)
            b64 = base64.b64encode(output_model).decode()
            href = f'<a href="data:file/output_model;base64,{b64}" download= model_lgbm.pkl >Download generated model (PKL File)</a>'
            st.markdown(href, unsafe_allow_html=True)

        download_model(opt_lgbm)