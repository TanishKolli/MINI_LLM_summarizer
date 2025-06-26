# Semantic Recursive Text Summarizer

A lightweight, interpretable, and efficient extractive text summarization tool using:
- **Sentence embeddings** (via `MiniLM`)
- **KMeans clustering**
- **spaCy + NLTK** for linguistic preprocessing
- **Recursive chunking** to handle long documents

---

##  Features

-  Handles **long inputs** via recursive chunking and summarization
-  Automatically detects **important & diverse sentences**
-  Uses **sentence embeddings** for semantic similarity
-  Supports basic **SVO extraction** (subject-verb-object) via `spaCy`
-  Works offline after model downloads
-  Returns human-readable, ordered summaries

---

## How it works

**Tokenization & Embedding**
- Sentences are tokenized and embedded using `MiniLM (all-MiniLM-L6-v2)`.

**Clustering**
- Sentences are grouped using `KMeans` to detect central ideas.

**Ranking**
- From each cluster, sentences are ranked using term frequency scores.

**Recursive Strategy**
- If the input is long (based on sentence count or token estimate), it is split into overlapping chunks, summarized individually, and finally re-summarized.

  ---

## What is MiniLM?

MiniLM (short for Miniature Language Model) is a compact transformer model from Microsoft that provides semantic sentence embeddings — numerical vectors that capture the meaning of sentences.
- Model used: `all-MiniLM-L6-v2`
- Size: ~22M parameters (fast + lightweight)
- Output: A 384-dimensional vector for each sentence
- Purpose: Maps similar sentences to nearby vectors in semantic space

  ---

  ## Step-by-Step Role of MiniLM
  - 1. **Sentence Tokenization** : The input text is split into individual sentences using NLTK.
    2. **Sentence Embedding with MiniLM** : Each sentence is passed through MiniLM to get a high-dimensional semantic vector
    3. **Semantic Clustering via KMeans** : These embeddings are grouped using KMeans clustering to detect central "topics" or "themes" across the document.
    4. **Centroid Sentence Selection** : From each cluster, the sentence closest to the center is chosen — this represents the "core idea" of that theme.
    5. **Ranking with Frequency (TF)** : To refine these, you apply frequency-based scoring and keep the most content-rich, diverse ones.
    6. **Optional Recursive Pass (if input is long)**:The same MiniLM is reused to embed partial summaries and summarize again.

---

## Why MiniLM Works So Well Here
- **Captures meaning beyond keywords** : Unlike bag-of-words or TF-IDF, MiniLM understands paraphrases and synonymy.
- **Speed & Efficiency** : MiniLM is extremely lightweight compared to BERT or RoBERTa but still highly accurate on sentence similarity tasks.
- **Plug-and-play with clustering** : Because it outputs fixed-length sentence vectors, it's easy to plug directly into KMeans, cosine similarity, etc.
