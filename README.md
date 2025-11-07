# 🎬 Persistent Self-Learning Hybrid Movie Recommender

This project is a **self-learning hybrid movie recommendation system** built using **Streamlit**, **PyTorch**, and **Sentence Transformers**. The system adapts and improves its recommendations over time by learning from user searches. Unlike static recommenders, this model **persists its learned knowledge across sessions** and fine-tunes itself based on user interactions.

---

## 🔹 Features

* **Hybrid Recommendation**: Combines semantic similarity using **Sentence-BERT embeddings** and item metadata (title, cast, genres, description).
* **Self-Learning**: Updates model weights after a configurable number of user searches to improve recommendations.
* **Persistent Storage**: Model weights and optimizer states are saved automatically so learning persists across Streamlit sessions.
* **Smart Search**:
    * Exact title match
    * Fuzzy typo correction
    * Semantic recommendations for movie themes or descriptions
* **Asynchronous Saving**: Model updates are saved in the background without blocking the UI.
* **Lightweight Online Training**: Trains on recent searches without requiring the full dataset every time.

## 🔹 File Structure
novel-recommendation-system/
│
├─ app.py                     # Main Streamlit app
├─ requirements.txt           # Python dependencies
├─ netflix_titles.csv         # Netflix movie dataset
├─ item_embeddings.npy        # Precomputed content embeddings
├─ hybrid_model.pt            # Saved model weights (auto-generated)
├─ optimizer.pt  # Saved optimizer state (auto-generated)
└─ README.md

## 🔹 How It Works
Data Loading: The CSV dataset is loaded, missing values are filled, and a combined content field is created for each movie.
Embeddings: Each movie's content is represented as a 384-dimensional embedding using SentenceTransformer("all-MiniLM-L6-v2").
Hybrid Model: Neural network combines Movie content embedding, Item ID embedding and User query embedding
Produces: Relevance score, Novelty score, Combines them to rank recommendations.
Smart Hybrid Search: Finds exact titles or close fuzzy matches, Computes semantic similarity with all movies, Combines title match scores and semantic scores to generate top recommendations.
Self-Learning Mechanism: The model finetunes itself after each search. Updates are saved asynchronously for persistent learning across sessions.

## 🔹 Usage
Run the Streamlit App
streamlit run app.py
Enter a movie title, theme, or keyword in the input box:
Example inputs: Inception, romantic drama, space adventure
Click "Get Recommendations".
The system will return the top recommended movies based on hybrid similarity.
The model learns from each search.

## 🔹 Model Persistence
Model weights (hybrid_model.pt) and optimizer state (optimizer.pt) are automatically saved after training.
When the app restarts, it loads the previous state to continue learning without losing knowledge.
The system performs incremental updates, so older knowledge is not erased — it’s updated gradually.

## 🔹 Configuration
Learning rate: Default is 0.0001 for Adam optimizer.
Embedding model: all-MiniLM-L6-v2 (can be replaced with other Sentence Transformers).

## 🔹 Dependencies
streamlit
pandas
numpy
torch
sentence-transformers
difflib

pip install -r requirements.txt

## 🔹 Notes
The app currently uses random positive and negative samples for lightweight online training. This is for demonstration purposes.
For a production-grade system, real user interaction data should be used to improve training quality.

## 🔹 License
This project is open-source and available under the MIT License.
