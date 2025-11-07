import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from difflib import get_close_matches, SequenceMatcher
import random
import os
import threading

# ---------------------------------------------------------
# 1️⃣ Streamlit Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Persistent Self-Learning Movie Recommender", layout="wide")

# ---------------------------------------------------------
# 2️⃣ Load Data and Models
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df.fillna("", inplace=True)
    df["content_text"] = (
        df["title"] + ". " +
        df["description"] + ". " +
        df["cast"] + ". " +
        df["listed_in"] + ". " +
        df["director"]
    )
    return df

@st.cache_resource
def load_embeddings():
    return np.load("item_embeddings.npy")

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------
# 3️⃣ Define Model
# ---------------------------------------------------------
class HybridNoveltyRecommender(nn.Module):
    def __init__(self, num_items, item_emb_dim=384, user_feat_dim=384, hidden=128):
        super().__init__()
        self.item_proj = nn.Linear(item_emb_dim, hidden)
        self.item_id_emb = nn.Embedding(num_items, hidden)
        self.user_proj = nn.Linear(user_feat_dim, hidden)

        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        self.novelty_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, item_content_emb, item_id, user_feat):
        c = F.relu(self.item_proj(item_content_emb))
        i = self.item_id_emb(item_id)
        u = F.relu(self.user_proj(user_feat))
        x = torch.cat([c, i, u], dim=-1)
        relevance = self.fusion(x).squeeze(-1)
        novelty = self.novelty_head(c).squeeze(-1)
        return relevance, novelty

# ---------------------------------------------------------
# 4️⃣ Smart Hybrid Search
# ---------------------------------------------------------
def smart_hybrid_search(query, df, item_emb_tensor, model_nn, model, device, alpha=0.7, topk_recommend=10):
    """
    Improved hybrid recommender:
    1. Shows exact title(s) first.
    2. Corrects typos and shows closest matches.
    3. Then recommends semantically similar movies without repeats.
    """
    query = query.strip().lower()
    used_titles = set()  # Track titles already added

    # Step 1: Exact/Partial match
    title_matches = df[df["title"].str.lower().str.contains(query, na=False)]
    seed_titles = []
    if not title_matches.empty:
        seed_titles = title_matches["title"].tolist()
    else:
        # Step 2: Fuzzy match for typos
        all_titles = df["title"].fillna("").tolist()
        close_titles = get_close_matches(query, all_titles, n=2, cutoff=0.6)
        if close_titles:
            seed_titles = close_titles

    # Step 3: Embedding for semantic recommendation
    search_text = query
    if seed_titles:
        search_text += " " + " ".join(seed_titles)
    search_emb = model.encode(search_text, convert_to_tensor=True).to(device)
    query_feat = search_emb.unsqueeze(0).repeat(len(df), 1)
    item_ids = torch.arange(len(df)).to(device)
    item_content = item_emb_tensor

    with torch.no_grad():
        relevance, novelty = model_nn(item_content, item_ids, query_feat)
        final_score = relevance + 0.5 * novelty
    semantic_scores = final_score.cpu().numpy()

    # Step 4: Combine semantic + fuzzy
    title_scores = np.array([
        SequenceMatcher(None, query, str(title).lower()).ratio()
        for title in df["title"]
    ])
    combined_scores = alpha * semantic_scores + (1 - alpha) * title_scores
    top_indices = np.argsort(combined_scores)[::-1]

    # Step 5: Build output
    result = []

    # Add exact/fuzzy matches first
    for s in seed_titles:
        matched_row = df[df["title"].str.lower() == s.lower()]
        if not matched_row.empty and s not in used_titles:
            result.append(matched_row.iloc[0])
            used_titles.add(s)

    # Add semantic recommendations, skipping already used titles
    for idx in top_indices:
        row = df.iloc[idx]
        if row["title"] not in used_titles:
            result.append(row)
            used_titles.add(row["title"])
        if len(result) >= topk_recommend:
            break

    final_df = pd.DataFrame(result).head(topk_recommend)
    return final_df, search_emb


# ---------------------------------------------------------
# 5️⃣ Online Training Step
# ---------------------------------------------------------
def online_train(model_nn, optimizer, search_emb, df, item_emb_tensor, num_samples=10, epochs=100):
    for epoch in range(epochs):
        model_nn.train()
        loss_fn = nn.BCEWithLogitsLoss()
        pos_ids = random.sample(range(len(df)), k=num_samples)
        neg_ids = random.sample(range(len(df)), k=num_samples)
        user_feat = search_emb.unsqueeze(0).repeat(num_samples * 2, 1)
        item_ids = torch.tensor(pos_ids + neg_ids, dtype=torch.long).to(device)
        item_content = item_emb_tensor[item_ids]
        labels = torch.cat([torch.ones(num_samples), torch.zeros(num_samples)]).to(device)

        relevance, novelty = model_nn(item_content, item_ids, user_feat)
        score = relevance + 0.5 * novelty
        loss = loss_fn(score, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

# ---------------------------------------------------------
# 6️⃣ Async Save Function
# ---------------------------------------------------------
def save_model_async(model, optimizer, model_file, optim_file):
    def _save():
        torch.save(model.state_dict(), model_file)
        torch.save(optimizer.state_dict(), optim_file)
    threading.Thread(target=_save, daemon=True).start()

# ---------------------------------------------------------
# 7️⃣ Load Data & Initialize
# ---------------------------------------------------------
df = load_data()
item_embeddings = load_embeddings()
model = load_sentence_model()

device = "cuda" if torch.cuda.is_available() else "cpu"
item_emb_tensor = torch.tensor(item_embeddings, dtype=torch.float32).to(device)
num_items = len(df)

MODEL_FILE = "hybrid_model.pt"
OPTIM_FILE = "optimizer.pt"

if os.path.exists(MODEL_FILE):
    st.session_state.model_nn = HybridNoveltyRecommender(num_items=num_items).to(device)
    st.session_state.model_nn.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    st.session_state.optimizer = torch.optim.Adam(st.session_state.model_nn.parameters(), lr=1e-4)
    if os.path.exists(OPTIM_FILE):
        st.session_state.optimizer.load_state_dict(torch.load(OPTIM_FILE, map_location=device))
    st.session_state.train_count = 0
else:
    st.session_state.model_nn = HybridNoveltyRecommender(num_items=num_items).to(device)
    st.session_state.optimizer = torch.optim.Adam(st.session_state.model_nn.parameters(), lr=1e-4)
    st.session_state.train_count = 0

# ---------------------------------------------------------
# 8️⃣ Streamlit UI
# ---------------------------------------------------------
st.title("🎬 Persistent Self-Learning Hybrid Movie Recommender")
st.markdown("Search for a movie — system learns from every search and saves knowledge asynchronously!")

query = st.text_input("🔍 Enter a movie title or theme:", placeholder="e.g., Inception, romantic drama, superhero")
if st.button("Get Recommendations") and query:
    with st.spinner("Generating recommendations..."):
        results, query_emb = smart_hybrid_search(
            query, df, item_emb_tensor, st.session_state.model_nn, model, device
        )

    # Online fine-tuning
    loss_val = online_train(
        st.session_state.model_nn, st.session_state.optimizer, query_emb, df, item_emb_tensor, 100
    )
    st.session_state.train_count += 1

    # Async save
    save_model_async(st.session_state.model_nn, st.session_state.optimizer, MODEL_FILE, OPTIM_FILE)

    st.success(f"✅ Model fine-tuned (iteration #{st.session_state.train_count}, loss={loss_val:.4f})")

    st.subheader("🎞 Recommended Titles:")
    for _, row in results.iterrows():
        with st.expander(f"**{row['title']}** ({row['release_year']}) — {row['listed_in']}"):
            st.write(row['description'])
else:
    st.info("Enter a movie title to start searching and training the model!")

st.markdown("---")
st.caption("This model updates itself online using each new search and saves asynchronously for persistent learning.")
