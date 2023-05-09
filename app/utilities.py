import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.util import community_detection
from sentence_transformers.util import semantic_search
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer


def embed_corpus(df,embedding_model):
    df['embedding'] = list(embedding_model.encode(df['content'].fillna("").to_list(), show_progress_bar=True))
    return df

def search_similar_desc_and_corresponding_solution_return(df,embedding_model, query: str, top_k: int = 3):
    desc = np.stack(df['embedding'].apply(lambda x: x.reshape(-1, len(x)))).squeeze(1)
    v_query = embedding_model.encode(query)
    similar_descriptions = semantic_search(v_query, desc , top_k=top_k)
    all_str=[]
    for i in range(top_k):
        df_index = similar_descriptions[0][i]["corpus_id"]
        s = df.iloc[df_index]
        str =f"{s['content']}".replace("\n", "") 
        all_str.append(str)

    return all_str