# Import libraries

import pandas as pd
from PIL import Image
import streamlit as st
import ml_algo
from pandas_profiling import ProfileReport
import avc_summary


#Set a title

st.title('DocHelper - AVC')

#Get the metadata

metadf = pd.read_csv('avc_metada.csv',sep=";")

#Set subheader and show data as table

st.subheader('Metadata')

st.dataframe(metadf)


# Get the previous overall data stats

overall_data = st.components.v1.html(avc_summary.report, width=700, height=350,scrolling=True)


# Get the feature input from the user
def get_user_input():

    Idade = st.sidebar.slider('Idade',0,100)
    Sexo = 	st.sidebar.slider('Sexo',0,1)
    Naturalidade = st.sidebar.slider('Naturalidade',0,1)	
    AVC_AIT_previo = st.sidebar.slider('AVC_AIT_previo',0,3,1)
    Antitrombóticos = st.sidebar.slider("Antitrombóticos",0,1)
    Dislip	= st.sidebar.slider("Dispip",0,1)
    Diabetes = st.sidebar.slider("Diabetes",0,2,1)
    HTA	 = st.sidebar.slider("HTA",0,1)
    Insuf_cardiaca = st.sidebar.slider("Insuf_cardiaca",0,1)
    EAM_prévio = st.sidebar.slider("EAM_prévio",0,2,1)
    FA = st.sidebar.slider("FA",0,2,1)
    Doenca_arterial_per	= st.sidebar.slider("Doenca_arterial_per",0,1)
    Tabagismo = st.sidebar.slider("Tabagismo",0,2,1)
    Alcool = st.sidebar.slider("Alcool",0,2,1)
    Estupefacientes = st.sidebar.slider("Estupefacientes",0,1,1)
    Peso = st.sidebar.slider("Peso",0,5,1)
    Dependencia_cognitiva = st.sidebar.slider("Dependencia_cognitiva",0,1)
    DPOC = st.sidebar.slider("DPOC",0,1)
    SAOS = st.sidebar.slider("SAOS",0,2,1)
    Hepatopatia = st.sidebar.slider("Hepatopatia",0,1)
    DRC = st.sidebar.slider("DRC",0,1)
    Neoplasia = st.sidebar.slider("Neoplasia",0,1)
    Depressão_distúrbio_ansiedade = st.sidebar.slider("Depressao_distúrbio_ansiedade",0,1)
    Incumprimento_terapêutico_recente = st.sidebar.slider("Incumprimento_terapêutico_recente",0,1)
    Anemia = st.sidebar.slider("Anemia",0,1)
    

    #Store dictionary into a variable

    user_data= {"Idade": Idade,
                "Sexo": Sexo,
                "Naturalidade": Naturalidade,
                "AVC_AIT_previo": AVC_AIT_previo,
                "Antitrombóticos": Antitrombóticos,
                "Dislip": Dislip,
                "Diabetes": Diabetes,
                "HTA": HTA,
                "Insuf_cardiaca": Insuf_cardiaca,
                "EAM_prévio": EAM_prévio,
                "FA": FA,
                "Doenca_arterial_per": Doenca_arterial_per,
                "Tagagismo": Tabagismo,
                "Alcool": Alcool,
                "Estupefacientes": Estupefacientes,
                "Peso": Peso,
                "Dependencia_cognitiva": Dependencia_cognitiva,
                "DPOC": DPOC,
                "SAOS": SAOS,
                "Hepatopatia": Hepatopatia,
                "DRC": DRC,
                "Neoplasia": Neoplasia,
                "Depressão_distúrbio_ansiedade": Depressão_distúrbio_ansiedade,
                "Incumprimento_terapêutico_recente": Incumprimento_terapêutico_recente,
                "Anemia": Anemia
                }

    #Data transformation into a dataframe

    features = pd.DataFrame(user_data, index=[0])
    return features

    



#Store users input into a variable

user_input = get_user_input()

# Set a subheader and display the users input

st.subheader("User Input:")
st.write(user_input)




#Feature importance

num_features = ml_algo.num_features

feature_importance = ml_algo.grid_search.best_estimator_.feature_importances_

FeatImp=pd.DataFrame(sorted(zip(feature_importance, num_features),reverse=True))

FeatImp.columns = ["Importance", "Features"]

#Show feature importance
st.subheader("Features importance:")

st.write(FeatImp)


#Show the model metrics
st.subheader("Model Teste Accuracy Score:")
st.write(str(ml_algo.Model_metrics*100)+"%")

#Store models predictions in a variable
prediction = ml_algo.grid_search.best_estimator_.predict(user_input)

#Set subheader and display classification
st.subheader("Classification:")
st.write(prediction)



