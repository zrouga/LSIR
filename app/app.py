import streamlit as st
import pandas as pd
import numpy as np
from utilities import *


st.set_page_config(
    page_title="Articles Information Retrieval Engine",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource(
    show_spinner="*Loading Articles ...*"
)
def load_data():
    df = pd.read_pickle('../data/api_data_embedded.pkl')
    return df

@st.cache_resource(
    show_spinner="*Loading Model ...*"
)
def load_model():
    return BERTopic.load('../data/topic_model_cyber.pkl')

@st.cache_resource(
    show_spinner="*Loading Embedding Model ...*"
)
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

with st.sidebar:
    st.subheader("About us :")
    st.markdown("Efficient InfoExtractor: A user-friendly GUI-based solution for on-demand information retrieval from a vast collection of articles. Seamlessly extract relevant data by simply prompting the system, saving time and effort while tackling information extraction challenges.")
    st.markdown("---")

data = load_data()
embedding_model = load_embedding_model()

_,col,_ = st.columns([0.5,2,0.5])
with col :
    st.title("Articles Info Retrieval Demo 🔍")
    st.write("")

_,col2,col3,col4= st.columns([2,4,1,2])
with col2 :
    st.write("")
    st.write("")
    user_query = st.text_input("",
        placeholder="Prompt your topic of interest and select k",
    )

with col3 :
    st.write("")
    st.write("")
    k_value = st.number_input('',min_value=1, max_value=15, value=3, step=1, key="k")

_,col2,col3= st.columns([1.2,1,1])
with col2 :
    query_button = st.button("Search For Articles",key="query_button")


articles = []
if query_button and user_query!='':
    articles = search_similar_desc_and_corresponding_solution_return(data,embedding_model,user_query,top_k=k_value)

    _,col2 = st.columns([0.5,4.5])
    with col2 :
        st.subheader("Retrieved Articles : ")
        for (i,article) in enumerate(articles) :
            st.markdown("Article " + str(i+1) + " :")
            st.write(article)
            st.markdown("---")
