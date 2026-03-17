import re
import random
from collections import Counter
from pathlib import Path

import numpy as np


def sigmoid(x):
    x = np.clip(x, -15, 15)
    return 1 / (1 + np.exp(-x))


def clean_and_split(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def make_vocab(tokens, min_count=2):
    counts = Counter(tokens)

    words = []
    for w, c in counts.items():
        if c >= min_count:
            words.append(w)

    word_to_idx = {w: i for i, w in enumerate(words)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    filtered = [w for w in tokens if w in word_to_idx]
    freqs = np.array([counts[idx_to_word[i]] for i in range(len(words))], dtype=np.int64)

    return filtered, word_to_idx, idx_to_word, freqs


def make_pairs(token_ids, window_size):
    pairs = []
    n = len(token_ids)

    for i in range(n):
        center_word = token_ids[i]

        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)

        for j in range(start, end):
            if j == i:
                continue
            context_word = token_ids[j]
            pairs.append((center_word, context_word))

    return pairs


class NegativeSampler:
    def __init__(self, counts):
        p = counts.astype(np.float64) ** 0.75
        p /= p.sum()
        self.p = p
        self.size = len(p)

    def sample(self, k, forbidden=None):
        if forbidden is None:
            forbidden = set()
        else:
            forbidden = set(forbidden)

        chosen = []
        while len(chosen) < k:
            draw = np.random.choice(self.size, size=2 * k, p=self.p)
            for x in draw:
                if x not in forbidden:
                    chosen.append(int(x))
                if len(chosen) == k:
                    break
        return chosen


class Word2Vec:
    def __init__(self, vocab_size, dim=50, lr=0.025, seed=42):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.dim = dim
        self.lr = lr

        self.W_in = rng.normal(0, 0.01, size=(vocab_size, dim))
        self.W_out = rng.normal(0, 0.01, size=(vocab_size, dim))

    def step(self, center_id, pos_id, neg_ids):
        u = self.W_in[center_id]
        v_pos = self.W_out[pos_id]
        v_neg = self.W_out[neg_ids]

        pos_score = v_pos @ u
        neg_score = v_neg @ u

        pos_prob = sigmoid(pos_score)
        neg_prob = sigmoid(neg_score)

        eps = 1e-10
        loss = -np.log(pos_prob + eps) - np.sum(np.log(1 - neg_prob + eps))

        pos_grad = pos_prob - 1
        neg_grad = neg_prob

        grad_u = pos_grad * v_pos + np.sum(neg_grad[:, None] * v_neg, axis=0)
        grad_v_pos = pos_grad * u
        grad_v_neg = neg_grad[:, None] * u[None, :]

        self.W_in[center_id] -= self.lr * grad_u
        self.W_out[pos_id] -= self.lr * grad_v_pos
        self.W_out[neg_ids] -= self.lr * grad_v_neg

        return float(loss)

    def embeddings(self):
        return self.W_in


def nearest_words(emb, word_to_idx, idx_to_word, query, top_k=5):
    if query not in word_to_idx:
        return []

    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    emb_norm = emb / norms

    q_idx = word_to_idx[query]
    q_vec = emb_norm[q_idx]

    scores = emb_norm @ q_vec
    order = np.argsort(-scores)

    out = []
    for idx in order:
        if idx == q_idx:
            continue
        out.append((idx_to_word[idx], float(scores[idx])))
        if len(out) >= top_k:
            break

    return out


def read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def main():
    data_path = "data/tiny_shakespeare.txt"
    min_count = 2
    window_size = 2
    dim = 50
    n_neg = 5
    epochs = 3
    lr = 0.025
    seed = 42
    print_every = 2000

    random.seed(seed)
    np.random.seed(seed)

    text = read_text(data_path)
    tokens = clean_and_split(text)
    print("Raw tokens:", len(tokens))

    tokens, word_to_idx, idx_to_word, counts = make_vocab(tokens, min_count=min_count)
    token_ids = [word_to_idx[w] for w in tokens]

    print("Tokens after filtering:", len(token_ids))
    print("Vocab size:", len(word_to_idx))

    pairs = make_pairs(token_ids, window_size)
    print("Number of training pairs:", len(pairs))

    sampler = NegativeSampler(counts)
    model = Word2Vec(len(word_to_idx), dim=dim, lr=lr, seed=seed)

    for epoch in range(epochs):
        random.shuffle(pairs)
        running_loss = 0.0

        for step_idx, (center_id, context_id) in enumerate(pairs, start=1):
            neg_ids = sampler.sample(n_neg, forbidden={context_id})
            loss = model.step(center_id, context_id, neg_ids)
            running_loss += loss

            if step_idx % print_every == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} - step {step_idx}/{len(pairs)} - "
                    f"avg loss {running_loss / print_every:.4f}"
                )
                running_loss = 0.0

        print(f"done with epoch {epoch + 1}")

    emb = model.embeddings()

    for word in ["king", "queen", "love", "man", "woman"]:
        if word not in word_to_idx:
            continue

        print(f"\nClosest words to '{word}':")
        for other_word, score in nearest_words(emb, word_to_idx, idx_to_word, word, top_k=5):
            print(f"  {other_word:15s} {score:.4f}")


if __name__ == "__main__":
    main()