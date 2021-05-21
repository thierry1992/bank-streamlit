#core pkgs
import streamlit as st
import streamlit.components.v1 as components
st.set_option('deprecation.showPyplotGlobalUse', False)
showPyplotGlobalUse = False

#EDA pkgs
import numpy as np
import pandas as pd
#import altair as alt
#Utils
import pickle
import lime
from lime import lime_tabular



#classifier model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error


#Data vizualisation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from PIL import Image




"""Banking credit app"""

image = Image.open('home-credit-bank-logo.jpeg')
st.image(image, use_column_width=True)
st.title("Credit Banking Prediction App")

menu = ["Accueil"]
submenu = ["Prediction", "Data Visualsiation"]

choice = st.sidebar.selectbox("Menu",menu)
if choice == "Accueil":
    st.subheader("Accueil")
    activity = st.sidebar.selectbox("Prediction", submenu)


    #Graphique    
if activity == "Prediction":
    st.subheader("Visualisation")
    def load_data():
        df = pd.read_csv("data/Data_Clean_Final.csv", index_col=0)
        #df = df.drop(columns=['target'])
        return df 
    df = load_data()      
     

    user_input = st.selectbox("ID",df.index.unique())

#Info Client    
    def info_client(user_input):   
        new_df_ID = df.loc[[user_input], :]
        #new_df_ID = pd.DataFrame(new_df_ID)
        st.write(new_df_ID.drop(columns=['target'])) 
        df_client = new_df_ID.drop(columns=['target'])
        return df_client
    client = info_client(user_input) 

    

    st.write("--")
    st.write("Prédiction ! Solvable / Non-Solvalble")

    def prediction(client):
        clf = pickle.load(open('Model/model.pkl', 'rb'))
        y_pred = clf.predict(client) 
        st.write(y_pred)
        X = df.drop(columns=['target'])

        #st.header("features importances")
        #plt.title('feature importance on Shap Values')
        #shap.summary_plot(shap_values, client)
        #st.pyplot(bbox_inches='tight')
        return y_pred

    class_label = {"Demande Refusée": 0, "Demande Acceptée" : 1}

    def get_key(val, my_dict):
        for key, value in my_dict.items():
            if val == value:
                return key    

    prediction = prediction(client)  
    
    final_result = get_key(prediction, class_label)
    st.success(final_result) 

    
    

    if st.button("Expliquer Resultat"):
        with st.spinner('Loading...'):
            def lime():
                df = load_data()
                X = df.drop(columns=['target'])
                y = df.target
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
                clf = pickle.load(open('Model/model.pkl', 'rb'))
                explainer = lime_tabular.LimeTabularExplainer(
                training_data=np.array(client),
                feature_names= client.columns,
                class_names=['Non Solvable', 'Solvable'],
                mode='classification'
                )
                exp = explainer.explain_instance(
                data_row= client.iloc[0], 
                predict_fn=clf.predict_proba, num_features= 7 
                )
                # Display explainer HTML object
                components.html(exp.as_html(), height=400 )
                return

            lime()  
            st.write("--")
            st.write("Features Importances ")
            clf = pickle.load(open('Model/model.pkl', 'rb'))
            feature_importance = clf.feature_importances_
            feature_importance = pd.DataFrame(feature_importance, index=client.columns, columns=['feature_importance'])
            feat = feature_importance.sort_values(by='feature_importance',ascending=False)[:10]
            fig = plt.figure()
            st.write(feat.plot(kind='bar'))
            st.pyplot()


            
                #all_columns = df_feat.columns.tolist()
                #primary_col = st.selectbox('Primary columns', all_columns)
                #feat_choice = st.multiselect("choisis une colone", all_columns)
                
                  

    
else:
    st.subheader("Data Visualsiation")  
    df = pd.read_csv("data/Data_Clean_Final.csv",index_col=0)
    if st.checkbox("Analyse de donnée"):
        number = st.number_input("Nombre de client à analyser", 5,10)
        st.dataframe(df.head(number))
        #user_input = st.selectbox("ID",df.index.unique())
        #st.write(df.loc[user_input,'ext_source_3'].plot(kind="bar"))

    user_input = st.selectbox("ID",df.index.unique())   

    def info_client(user_input):   
        new_df_ID = df.loc[[user_input], :]
        #new_df_ID = pd.DataFrame(new_df_ID)
        st.write(new_df_ID.drop(columns=['target'])) 
        df_client = new_df_ID.drop(columns=['target'])
        return df_client
        
    client = info_client(user_input) 

    if st.checkbox("Analyse de la target"):
        
                features_import = ['ext_source_1',
                'ext_source_3',
                'ext_source_2',
                'payment_rate',
                'annuity_income_perc',
                'amt_credit',  
                'amt_goods_price',                 
                'days_employed_perc',
                'amt_annuity',
                'amt_income_total',                
                'target']
                importance_feat = pd.DataFrame(df[features_import])
                solvable = importance_feat[importance_feat['target'] == 1]
                non_solvable = importance_feat[importance_feat['target'] == 0]
                feat_choice = st.selectbox('Choisissez une features', features_import)
                fig = plt.figure()
                plt.axvline(x=client[feat_choice].values, c='r')
                sns.kdeplot(solvable[feat_choice],label='solvable')
                sns.kdeplot(non_solvable[feat_choice],label='non solvable')
                plt.legend()
                st.pyplot(fig)    









         
        
                    
    
    
    




        
    

                  



