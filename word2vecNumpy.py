import re
import random
from collections import Counter
from pathlib import Path

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    x = np.clip(x, -15, 15)
    return 1.0 / (1.0 + np.exp(-x))


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def build_vocab(tokens, min_count=2):
    counter = Counter(tokens)
    vocab_words = [w for w, c in counter.items() if c >= min_count]
    word_to_id = {w: i for i, w in enumerate(vocab_words)}
    id_to_word = {i: w for w, i in word_to_id.items()}

    filtered_tokens = [w for w in tokens if w in word_to_id]
    word_counts = np.array([counter[id_to_word[i]] for i in range(len(vocab_words))], dtype=np.int64)

    return filtered_tokens, word_to_id, id_to_word, word_counts


def generate_skipgram_pairs(token_ids, window_size):
    pairs = []
    n = len(token_ids)
    for center_pos in range(n):
        center = token_ids[center_pos]
        left = max(0, center_pos - window_size)
        right = min(n, center_pos + window_size + 1)
        for ctx_pos in range(left, right):
            if ctx_pos == center_pos:
                continue
            context = token_ids[ctx_pos]
            pairs.append((center, context))
    return pairs


class UnigramSampler:
    """
    Negative sampler using the word2vec unigram distribution raised to 0.75.
    """
    def __init__(self, counts):
        probs = counts.astype(np.float64) ** 0.75
        probs /= probs.sum()
        self.probs = probs
        self.vocab_size = len(probs)

    def sample(self, k, forbidden=None):
        result = []
        forbidden = set() if forbidden is None else set(forbidden)
        while len(result) < k:
            samples = np.random.choice(self.vocab_size, size=k * 2, p=self.probs)
            for s in samples:
                if s not in forbidden:
                    result.append(int(s))
                if len(result) == k:
                    break
        return result


class Word2VecSGNS:
    def __init__(self, vocab_size, embed_dim=50, lr=0.025, seed=42):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = lr

        # Input embeddings (center words)
        self.W_in = rng.normal(0.0, 0.01, size=(vocab_size, embed_dim))
        # Output embeddings (context words)
        self.W_out = rng.normal(0.0, 0.01, size=(vocab_size, embed_dim))

    def train_step(self, center_id, pos_context_id, neg_context_ids):
        """
        Skip-gram with negative sampling.

        For one center word c, one positive context o, and K negatives n_k:
            L = -log sigma(v_o^T u_c) - sum_k log sigma(-v_nk^T u_c)

        Returns scalar loss.
        """

        u = self.W_in[center_id]                 # shape: (D,)
        v_pos = self.W_out[pos_context_id]       # shape: (D,)
        v_negs = self.W_out[neg_context_ids]     # shape: (K, D)

        # Forward
        pos_score = np.dot(v_pos, u)             # scalar
        neg_scores = np.dot(v_negs, u)           # shape: (K,)

        pos_sig = sigmoid(pos_score)             # sigma(v_o^T u_c)
        neg_sig = sigmoid(neg_scores)            # sigma(v_n^T u_c)

        # Loss
        eps = 1e-10
        loss_pos = -np.log(pos_sig + eps)
        loss_neg = -np.sum(np.log(1.0 - neg_sig + eps))
        loss = loss_pos + loss_neg

        # Gradients
        # d/dx[-log sigma(x)] = sigma(x) - 1
        g_pos = pos_sig - 1.0                    # scalar

        # d/dx[-log sigma(-x)] = sigma(x)
        g_negs = neg_sig                         # shape: (K,)

        # Grad wrt input embedding u
        grad_u = g_pos * v_pos + np.sum(g_negs[:, None] * v_negs, axis=0)

        # Grad wrt positive output embedding
        grad_v_pos = g_pos * u

        # Grad wrt negative output embeddings
        grad_v_negs = g_negs[:, None] * u[None, :]

        # SGD updates
        self.W_in[center_id] -= self.lr * grad_u
        self.W_out[pos_context_id] -= self.lr * grad_v_pos
        self.W_out[neg_context_ids] -= self.lr * grad_v_negs

        return float(loss)

    def get_embeddings(self):
        # Often the final embedding is W_in or (W_in + W_out)/2
        return self.W_in


def most_similar(embeddings, word_to_id, id_to_word, query_word, top_k=5):
    if query_word not in word_to_id:
        return []

    E = embeddings
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-10
    E_norm = E / norms

    qid = word_to_id[query_word]
    qvec = E_norm[qid]
    sims = E_norm @ qvec
    best = np.argsort(-sims)

    result = []
    for idx in best:
        if idx == qid:
            continue
        result.append((id_to_word[idx], float(sims[idx])))
        if len(result) == top_k:
            break
    return result


def load_text_file(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def main():
    # --------------------------
    # Config
    # --------------------------
    data_path = "data/tiny_shakespeare.txt"
    min_count = 2
    window_size = 2
    embed_dim = 50
    neg_samples = 5
    epochs = 3
    lr = 0.025
    seed = 42
    print_every = 2000

    random.seed(seed)
    np.random.seed(seed)

    # --------------------------
    # Load + preprocess
    # --------------------------
    text = load_text_file(data_path)
    tokens = tokenize(text)
    print(f"Total raw tokens: {len(tokens)}")

    tokens, word_to_id, id_to_word, word_counts = build_vocab(tokens, min_count=min_count)
    token_ids = [word_to_id[w] for w in tokens]
    vocab_size = len(word_to_id)

    print(f"Filtered tokens: {len(token_ids)}")
    print(f"Vocab size: {vocab_size}")

    # --------------------------
    # Training data
    # --------------------------
    pairs = generate_skipgram_pairs(token_ids, window_size=window_size)
    print(f"Training pairs: {len(pairs)}")

    sampler = UnigramSampler(word_counts)
    model = Word2VecSGNS(vocab_size=vocab_size, embed_dim=embed_dim, lr=lr, seed=seed)

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0.0

        for step, (center_id, pos_context_id) in enumerate(pairs, start=1):
            neg_ids = sampler.sample(neg_samples, forbidden={pos_context_id})
            loss = model.train_step(center_id, pos_context_id, neg_ids)
            total_loss += loss

            if step % print_every == 0:
                avg_loss = total_loss / print_every
                print(f"Epoch {epoch+1}/{epochs} | Step {step}/{len(pairs)} | Avg Loss {avg_loss:.4f}")
                total_loss = 0.0

        print(f"Finished epoch {epoch+1}/{epochs}")

    # --------------------------
    # Inspect embeddings
    # --------------------------
    embeddings = model.get_embeddings()
    test_words = ["king", "queen", "love", "man", "woman"]

    for w in test_words:
        if w in word_to_id:
            sims = most_similar(embeddings, word_to_id, id_to_word, w, top_k=5)
            print(f"\nMost similar to '{w}':")
            for word, score in sims:
                print(f"  {word:15s} {score:.4f}")


if __name__ == "__main__":
    main()