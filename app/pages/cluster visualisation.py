import streamlit as st
from bertopic import BERTopic
import plotly.express as px

st.set_page_config(
    page_title="Articles Information Retrieval Engcdcdscdine",
    page_icon="🔍",
    layout="wide"
)

with st.sidebar:
    st.subheader("About us :")
    st.markdown("Efficient InfoExtractor: A user-friendly GUI-based solution for on-demand information retrieval from a vast collection of articles. Seamlessly extract relevant data by simply prompting the system, saving time and effort while tackling information extraction challenges.")
    st.markdown("---")

@st.cache_resource(
    show_spinner="*Loading Model ...*"
)
def load_model():
    return BERTopic.load('../data/topic_model_cyber.pkl')

def plot_intertopic(topic_model):
    fig = topic_model.visualize_topics()
    return fig

def plot_hierarchy(topic_model):
    return topic_model.visualize_hierarchy()

def plot_term_topic(topic_model):
    return topic_model.visualize_term_rank()

def plot_heat_map(topic_model):
    return topic_model.visualize_heatmap()

def plot_barchart(topic_model):
    return topic_model.visualize_barchart()

_,col,_ = st.columns([0.5,2,0.5])
with col :
    
    st.title("Clusters and Data Visualization")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

model = load_model()
intertopic_figure = plot_intertopic(model)
hierarchy_figure = plot_hierarchy(model)
termtopic_figure = plot_term_topic(model)
heatmap_figure = plot_heat_map(model)
barchart_figure = plot_barchart(model)

_,col2,col3,col4= st.columns([0.1,3,2,1])
with col2 :
    st.write("")
    st.write("")
    st.subheader("Please select the aspect to visualize :")
with col3 :
    option = st.selectbox(
        '',
        ('Nothing Selected', 'Inter Topic Distance Plot', 'HDBSCAN Hierarchy', 'Term Score Decline Per Topic','Bar Chart', 'Heat Map'))

col1,col2 = st.columns([0.1,9.9])
with col2:
    if option == 'Inter Topic Distance Plot' :
        st.plotly_chart(intertopic_figure,use_width_container=True)
    elif option == 'HDBSCAN Hierarchy' :
        st.plotly_chart(hierarchy_figure,use_width_container=True)
    elif option == 'Term Score Decline Per Topic' :
        st.plotly_chart(termtopic_figure,use_width_container=True)
    elif option == 'Bar Chart':
        st.plotly_chart(barchart_figure,use_width_container=True)
    elif option == 'Heat Map':
        st.plotly_chart(heatmap_figure,use_width_container=True)