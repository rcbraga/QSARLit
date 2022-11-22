import streamlit as st
from multiapp import MultiApp
import home, cur, cur_vs, desc, rf, svm, lgbm, vs, maps, rf_re, svm_re, lgbm_re #import your app modules here
import utils

#utils.deploy_chembl()
app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Curation for modeling", cur.app)
#app.add_app("Curation for Virtual Screening", cur_vs.app)
app.add_app("Calculate Descriptors", desc.app)
app.add_app("Random Forest - Classification", rf.app)
app.add_app("Support Vector Classification", svm.app)
app.add_app("LightGBM - Classification", lgbm.app)
app.add_app("Random Forest - Regressor", rf_re.app)
app.add_app("Support Vector Regressor", svm_re.app)
app.add_app("LightGBM - Regressor", lgbm_re.app)
app.add_app("Virtual Screening", vs.app)
app.add_app("Probability Maps", maps.app)
cc = utils.Custom_Components()

# The main app
s_state = st.session_state
if 'title' not in s_state:
    s_state["title"] = {"title":'Home',"function":home.app}
    s_title = s_state["title"]["title"]
    st.write(s_state)
    if s_title == 'Home':
        s_state.df = None
        app.run(None,s_state)
    else:
        #s_state = st.session_state
        s_state.df = cc.upload_file()
        app.run(s_state.df,s_state)
else:
        st.write(s_state)
        s_title = s_state["title"]["title"]
        if s_title == 'Home':
            s_state.df = None
            app.run(None,s_state)
        else:
            #s_state = st.session_state
            s_state.df = cc.upload_file()
            app.run(s_state.df,s_state)
