#if "updated_df" not in st.session_state:
    #         st.session_state.updated_df = df
        
    #         st.header('**Original input data**')
    #         AgGrid(df)
   
    #     st.sidebar.header("Please delete undesired columns")
        
    #     with st.sidebar.form("my_form"):
    #         index = df.columns.tolist().index(
    #             st.session_state["updated_df"].columns.tolist()[0]
    #         )
    #         st.selectbox(
    #             "Select column to delete", options=df.columns, index=index, key="delete_col"
    #         )
    #         delete = st.form_submit_button(label="Delete")
    #     if delete:
    #         persist_dataframe("updated_df","delete_col")
    # else:
    #     st.info('Awaiting for CSV file to be uploaded.')
# with st.sidebar.header('2. Upload your CSV data (calculated descriptors)'):
#         uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
#     st.sidebar.markdown("""
#     [Example CSV input file](https://github.com/joseteofilo/data_qsarlit/blob/master/descriptor_morgan_r2_2048bits_for_modeling.csv)
#     """)

#     # Read Uploaded file and convert to pandas
#     if uploaded_file is not None:
#         # Read CSV data
#         df = pd.read_csv(uploaded_file, sep=',')

#         st.header('**Molecular descriptors input data**')

#         AgGrid(df)

#     else:
#         st.info('Awaiting for CSV file to be uploaded.')

#     st.sidebar.write('---')
# with st.sidebar.header('2. Upload your CSV data (calculated descriptors)'):
#         uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
#     st.sidebar.markdown("""
#     [Example CSV input file](https://github.com/joseteofilo/data_qsarlit/blob/master/descriptor_morgan_r2_2048bits_for_modeling.csv)
#     """)

#     # Read Uploaded file and convert to pandas
#     if uploaded_file is not None:
#         # Read CSV data
#         df = pd.read_csv(uploaded_file, sep=',')

#         st.header('**Molecular descriptors input data**')

#         AgGrid(df)

#     else:
#         st.info('Awaiting for CSV file to be uploaded.')

#     st.sidebar.write('---')
