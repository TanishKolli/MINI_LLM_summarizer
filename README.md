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
-Sentences are tokenized and embedded using `MiniLM (all-MiniLM-L6-v2)`.

**Clustering**
-Sentences are grouped using `KMeans` to detect central ideas.

**Ranking**
-From each cluster, sentences are ranked using term frequency scores.

**Recursive Strategy**
-If the input is long (based on sentence count or token estimate), it is split into overlapping chunks, summarized individually, and finally re-summarized.
