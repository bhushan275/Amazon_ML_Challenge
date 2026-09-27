"""
Stage 2: Blocking & Candidate Generation Module
Implements multi-pass fuzzy blocking:
- Pass 1: Token & character n-gram inverted index
- Pass 2: First-token + location/postal key blocking
- Pass 3: Sound-alike / phonetic key blocking
- Pass 4: TF-IDF Cosine Similarity K-Nearest Neighbor search

Unions candidate sets per S1 entity to establish high recall ceiling while reducing search space.
Outputs candidate_pairs.tsv per specification.
"""

import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def generate_soundex_key(text: str) -> str:
    """Generates a simplified phonetic soundex-like key for a token."""
    if not text:
        return ""
    text = text.upper()
    first_char = text[0]
    # Simple soundex mapping table
    char_map = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    encoded = [first_char]
    for char in text[1:]:
        code = char_map.get(char, '0')
        if code != '0' and code != encoded[-1]:
            encoded.append(code)
    soundex = ("".join(encoded) + "0000")[:4]
    return soundex


class MultiPassBlocker:
    def __init__(self, top_k_tfidf: int = 15, min_ngram_overlap: int = 2):
        self.top_k_tfidf = top_k_tfidf
        self.min_ngram_overlap = min_ngram_overlap

    def fit_transform(self, s1_records: list, target_records: list) -> dict:
        """
        Takes preprocessed S1 records and target (S2 + S3) records.
        Returns candidate map: s1_id -> list of candidate target_ids.
        """
        candidates = collections.defaultdict(set)

        # Build Inverted Indexes for target (S2/S3) records
        token_index = collections.defaultdict(list)
        phonetic_index = collections.defaultdict(list)
        first_word_index = collections.defaultdict(list)

        target_ids = []
        target_texts = []

        for idx, rec in enumerate(target_records):
            t_id = rec['entity_id']
            name_clean = rec['name_clean']
            addr_clean = rec['address_clean']
            country = rec['country']

            target_ids.append(t_id)
            target_texts.append(f"{name_clean} {addr_clean} {country}".strip())

            # 1. Token index
            tokens = name_clean.split()
            for tok in tokens:
                if len(tok) >= 3:
                    token_index[tok].append(t_id)
                    # 2. Phonetic index
                    phone_key = generate_soundex_key(tok)
                    if phone_key:
                        phonetic_index[phone_key].append(t_id)

            # 3. First word + Country index
            if tokens:
                first_word_key = f"{tokens[0]}_{country}"
                first_word_index[first_word_key].append(t_id)

        # --- Pass 1, 2, 3: Rule-based & Phonetic Inverted Index Queries ---
        for s1_rec in s1_records:
            s1_id = s1_rec['entity_id']
            s1_name = s1_rec['name_clean']
            s1_country = s1_rec['country']
            s1_tokens = s1_name.split()

            # Token overlap pass
            token_counts = collections.Counter()
            for tok in s1_tokens:
                if len(tok) >= 3:
                    for matched_t_id in token_index.get(tok, []):
                        token_counts[matched_t_id] += 1
                    # Phonetic pass
                    phone_key = generate_soundex_key(tok)
                    for matched_t_id in phonetic_index.get(phone_key, []):
                        token_counts[matched_t_id] += 1

            for t_id, cnt in token_counts.items():
                if cnt >= self.min_ngram_overlap:
                    candidates[s1_id].add(t_id)

            # First word + Country pass
            if s1_tokens:
                fw_key = f"{s1_tokens[0]}_{s1_country}"
                for matched_t_id in first_word_index.get(fw_key, []):
                    candidates[s1_id].add(matched_t_id)

        # --- Pass 4: TF-IDF Cosine Similarity Nearest Neighbors ---
        if target_texts and s1_records:
            s1_texts = [f"{r['name_clean']} {r['address_clean']} {r['country']}".strip() for r in s1_records]

            tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            target_matrix = tfidf.fit_transform(target_texts)
            s1_matrix = tfidf.transform(s1_texts)

            n_neighbors = min(self.top_k_tfidf, target_matrix.shape[0])
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
            nn.fit(target_matrix)

            distances, indices = nn.kneighbors(s1_matrix)

            for s1_idx, neighbor_indices in enumerate(indices):
                s1_id = s1_records[s1_idx]['entity_id']
                for n_idx in neighbor_indices:
                    cand_id = target_ids[n_idx]
                    candidates[s1_id].add(cand_id)

        # Format output mapping: s1_id -> sorted list of candidate IDs
        result = {}
        for s1_rec in s1_records:
            s1_id = s1_rec['entity_id']
            cand_list = sorted(list(candidates.get(s1_id, set())))
            result[s1_id] = cand_list

        return result


def compute_blocking_metrics(candidate_map: dict, ground_truth: dict, total_target_count: int):
    """
    Computes Recall Ceiling and Reduction Ratio for blocking.
    """
    total_true_pairs = 0
    retained_true_pairs = 0
    total_candidates_generated = 0

    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        total_true_pairs += len(true_matches)
        generated_cands = set(candidate_map.get(s1_id, []))
        total_candidates_generated += len(generated_cands)

        retained_true_pairs += len(set(true_matches) & generated_cands)

    recall_ceiling = retained_true_pairs / max(total_true_pairs, 1)
    total_possible_pairs = len(candidate_map) * total_target_count
    reduction_ratio = 1.0 - (total_candidates_generated / max(total_possible_pairs, 1))

    return {
        "recall_ceiling": recall_ceiling,
        "reduction_ratio": reduction_ratio,
        "total_candidates": total_candidates_generated,
        "retained_true_pairs": retained_true_pairs,
        "total_true_pairs": total_true_pairs
    }
