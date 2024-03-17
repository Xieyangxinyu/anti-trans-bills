import pandas as pd
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import openai
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech
from bertopic import BERTopic
import os
import pickle
import dill

def get_topic_model(df, bill_type: str = 'Healthcare'):
    # check if 'data/{bill_type} exists
    if os.path.exists(f'data/{bill_type}'):
        docs = pickle.load(open(f'data/{bill_type}/docs.pkl', 'rb'))
        topic_model = dill.load(open(f'data/{bill_type}/topic_model.pkl', 'rb'))
        embeddings = pickle.load(open(f'data/{bill_type}/embeddings.pkl', 'rb'))
        topic_df = pd.read_csv(f'data/{bill_type}/topic.csv')
        info_tab = pd.read_csv(f'data/{bill_type}/topic_info.csv')
    else:
        os.makedirs(f'data/{bill_type}')
        docs = df[df['Bill Type'] == bill_type]['Bill Description']
        #sentences = [sent_tokenize(doc) for doc in docs]
        #docs = [sentence for doc in sentences for sentence in doc]
        docs = pd.Series(docs).unique()
        
        embedding_model = SentenceTransformer("all-mpnet-base-v2")
        embeddings = embedding_model.encode(docs, show_progress_bar=True)
        umap_model = UMAP(n_neighbors=min(15, int(len(docs)/5)), n_components=5, min_dist=0.1, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
        keybert_model = KeyBERTInspired()
        pos_model = PartOfSpeech("en_core_web_sm")
        mmr_model = MaximalMarginalRelevance(diversity=0.3)

        # GPT-3.5

        client = openai.OpenAI(api_key="sk-*******")

        prompt = """
        I have a topic that contains the following documents:
        [DOCUMENTS]

        Based on the information above, extract an informative topic label of at most 10 words. Stick to the original language of the document as much as possible and avoid mentioning of locations. Make sure it is in the following format:
        topic: <topic label>
        """
        openai_model = OpenAI(client, model="gpt-4-0125-preview", exponential_backoff=True, chat=True, prompt=prompt)

        # All representation models
        representation_model = {
            "KeyBERT": keybert_model,
            "OpenAI": openai_model,  # Uncomment if you will use OpenAI
            "MMR": mmr_model,
            "POS": pos_model
        }


        topic_model = BERTopic(
            # Pipeline models
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            # Hyperparameters
            top_n_words=10,
            verbose=True
        )

        topics, _ = topic_model.fit_transform(docs, embeddings)
        chatgpt_topic_labels = {topic: " | ".join(list(zip(*values))[0]) for topic, values in topic_model.topic_aspects_["OpenAI"].items()}
        chatgpt_topic_labels[-1] = "Outlier Topic"
        topic_model.set_topic_labels(chatgpt_topic_labels)

        topic_model.representation_model = {
            "KeyBERT": keybert_model,
            "MMR": mmr_model,
            "POS": pos_model
        }

        info_tab = topic_model.get_topic_info()
        topic_df = pd.DataFrame(columns=['Topic', 'Documents'])
        topic_df['Topic'] = topics
        topic_df['Documents'] = docs
        topic_df.to_csv(f'data/{bill_type}/topic.csv', index=False)
        info_tab.to_csv(f'data/{bill_type}/topic_info.csv', index=False)
        pickle.dump(docs, open(f'data/{bill_type}/docs.pkl', 'wb'))
        dill.dump(topic_model, open(f'data/{bill_type}/topic_model.pkl', 'wb'))
        pickle.dump(embeddings, open(f'data/{bill_type}/embeddings.pkl', 'wb'))
    return docs, embeddings, topic_model, topic_df, info_tab


