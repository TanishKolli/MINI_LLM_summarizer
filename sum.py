import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter
import spacy
# Setup
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

# Load small semantic transformer
sim_model = SentenceTransformer("all-MiniLM-L6-v2")

from sentence_transformers import SentenceTransformer
sim_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(sentences):
    return sim_model.encode(sentences)

# --- Utility Functions ---
def estimate_token_count(text):
    return int(len(word_tokenize(text)) * 1.3)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def score_sentence(sentence, word_freq):
    words = [w.lower() for w in word_tokenize(sentence) if w.isalpha() and w not in stop_words]
    return sum(word_freq.get(w, 0) for w in words)

def clean_sentence(s):
    s = s.strip()
    return s[0].upper() + s[1:] if s else s

def extract_svo(sentence):
    doc = nlp(sentence)
    subj, verb, obj = None, None, None
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            subj = token.text
        elif token.pos_ == "VERB":
            verb = token.text
        elif token.dep_ in ("dobj", "pobj", "attr"):
            obj = token.text
    return subj, verb, obj

# --- Unified Summarizer ---
def summarize(text, top_n=7, chunk_sent_limit=15, overlap=2, token_threshold=512):
    sentences = sent_tokenize(text)
    token_count = estimate_token_count(text)

    # Recursive chunking if text is too long
    if token_count > token_threshold or len(sentences) > chunk_sent_limit * 2:
        chunks = []
        i = 0
        while i < len(sentences):
            chunk = sentences[i:i + chunk_sent_limit]
            chunks.append(" ".join(chunk))
            i += chunk_sent_limit - overlap

        partial_summaries = [summarize(chunk, top_n=top_n) for chunk in chunks]
        final_input = " ".join(partial_summaries)
        return summarize(final_input, top_n=max(3, top_n // 2))  # Shrink top_n on recursion

    # Normal summarization
    if len(sentences) == 0:
        return ""

    embeddings = sim_model.encode(sentences)
    n = len(sentences)
    est_k = max(3, min(n, token_count // 60))
    Ks = list(range(est_k, min(est_k + 4, n)))

    inertias = []
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=0).fit(embeddings)
        inertias.append(km.inertia_)
    deltas = np.diff(inertias)
    k = Ks[np.argmin(deltas)] if len(deltas) > 0 else Ks[0]

    kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
    centers = kmeans.cluster_centers_

    candidate_indices = []
    for center in centers:
        idx = np.argmin([np.linalg.norm(center - emb) for emb in embeddings])
        candidate_indices.append(idx)

    word_freq = Counter([
        w.lower() for sent in sentences for w in word_tokenize(sent)
        if w.isalpha() and w not in stop_words
    ])
    ranked = sorted(candidate_indices, key=lambda i: score_sentence(sentences[i], word_freq), reverse=True)
    selected = sorted(ranked[:top_n])

    summary_sentences = [clean_sentence(sentences[i]) for i in selected]
    return " ".join(summary_sentences)

