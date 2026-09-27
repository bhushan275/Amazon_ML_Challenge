"""
Core Blocking & Candidate Generation Module (P2 Ownership)
===========================================================
Implements high-recall, high-reduction multi-pass blocking strategies:
1. Inverted Index with Frequency Filtering (rare/informative token indexing)
2. Phonetic Key Indexing (Soundex & Simplified Metaphone)
3. Country-Aware Composite Blocking Keys
4. Sorted Neighborhood Method (SNM) with multi-key sliding windows
5. MinHash & Locality-Sensitive Hashing (LSH) for sub-linear fuzzy matching
6. TF-IDF Cosine Similarity Top-K Neighbor Retrieval (with stdlib fallback)

Optimized for:
- Recall Ceiling (> 98%)
- Reduction Ratio (> 99%)
- Strict adherence to challenge rules: S2/S3 IDs only, no S1 self-matches.
"""

import re
import math
import hashlib
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any


# =====================================================================
# 1. Phonetic & Sound-Alike Hashing
# =====================================================================

def generate_soundex_key(text: str) -> str:
    """
    Computes classic American Soundex code for a token.
    Standardized to 4 characters: [Letter][Digit][Digit][Digit].
    """
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'[^A-Za-z]', '', text).upper()
    if not text:
        return ""

    first_char = text[0]
    char_map = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }

    digits = [first_char]
    prev_code = char_map.get(first_char, '0')

    for char in text[1:]:
        code = char_map.get(char, '0')
        if code != '0':
            if code != prev_code:
                digits.append(code)
            prev_code = code
        else:
            prev_code = '0'

    soundex_code = "".join(digits)
    # Pad with zeros or truncate to 4 chars
    return (soundex_code + "0000")[:4]


def generate_metaphone_key(text: str) -> str:
    """
    Computes a simplified phonetic metaphone-style representation.
    Maps similar sounding consonants and collapses duplicate phonemes.
    """
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z]', '', text).lower()
    if not text:
        return ""

    # Common English / European phonetic transformations
    text = re.sub(r'^kn|^gn|^pn|^wr', lambda m: m.group(0)[1], text)
    text = re.sub(r'ph', 'f', text)
    text = re.sub(r'sh|ch', 'x', text)
    text = re.sub(r'ck|qu|q', 'k', text)
    text = re.sub(r'c(?=[eiy])', 's', text)
    text = re.sub(r'c', 'k', text)
    text = re.sub(r'dg', 'j', text)
    text = re.sub(r'th', '0', text)
    text = re.sub(r'w(?![aeiou])', '', text)
    text = re.sub(r'z', 's', text)
    text = re.sub(r'v', 'f', text)

    # Collapse repeated adjacent characters
    out = [text[0]]
    for ch in text[1:]:
        if ch != out[-1]:
            out.append(ch)

    # Keep first vowel, drop subsequent interior vowels
    vowels = set('aeiou')
    result = [out[0].upper()]
    for ch in out[1:]:
        if ch not in vowels:
            result.append(ch.upper())

    return "".join(result)[:6]


# =====================================================================
# 2. Country-Aware Blocking
# =====================================================================

class CountryAwareBlocker:
    """
    Generates multi-attribute composite blocking keys conditioned on country.
    Includes fallback keys for open-set / noisy country values.
    """

    def __init__(self, max_bucket_size: int = 500):
        self.max_bucket_size = max_bucket_size
        self.buckets = defaultdict(list)

    @staticmethod
    def extract_blocking_keys(record: dict) -> List[str]:
        keys = []
        name = record.get('name_clean', '') or record.get('business_name', '')
        addr = record.get('address_clean', '') or record.get('business_address', '')
        country = (record.get('country', '') or 'UNKNOWN').upper().strip()

        tokens = [tok for tok in name.lower().split() if len(tok) >= 2]
        addr_tokens = [tok for tok in addr.lower().split() if len(tok) >= 2]
        numeric_tokens = [tok for tok in addr_tokens if tok.isdigit()]

        if not tokens:
            return [f"GEN_{country}"]

        # Key 1: First token + Country
        keys.append(f"FW_{tokens[0]}_{country}")

        # Key 2: Soundex of first token + Country
        snd1 = generate_soundex_key(tokens[0])
        if snd1:
            keys.append(f"SND_{snd1}_{country}")

        # Key 3: Metaphone of first token + Country
        meta1 = generate_metaphone_key(tokens[0])
        if meta1:
            keys.append(f"META_{meta1}_{country}")

        # Key 4: First two tokens sorted + Country
        if len(tokens) >= 2:
            sorted_pair = "_".join(sorted(tokens[:2]))
            keys.append(f"PAIR_{sorted_pair}_{country}")

        # Key 5: Numeric address token + Country (e.g., postal code / building number)
        for num in numeric_tokens[:2]:
            keys.append(f"NUM_{num}_{country}")

        # Key 6: Country-agnostic fallback (handles country mismatch / open-set France/India/US)
        keys.append(f"FW_ANY_{tokens[0]}")
        if len(tokens) >= 2:
            keys.append(f"PAIR_ANY_{tokens[0]}_{tokens[1]}")

        return keys

    def fit(self, target_records: List[dict]):
        self.buckets.clear()
        for rec in target_records:
            t_id = rec['entity_id']
            for k in self.extract_blocking_keys(rec):
                self.buckets[k].append(t_id)

    def query(self, s1_records: List[dict]) -> Dict[str, Set[str]]:
        candidates = defaultdict(set)
        for rec in s1_records:
            s1_id = rec['entity_id']
            for k in self.extract_blocking_keys(rec):
                bucket_targets = self.buckets.get(k, [])
                if 0 < len(bucket_targets) <= self.max_bucket_size:
                    for t_id in bucket_targets:
                        candidates[s1_id].add(t_id)
        return candidates


# =====================================================================
# 3. Sorted Neighborhood Method (SNM)
# =====================================================================

class SortedNeighborhoodBlocker:
    """
    Sorted Neighborhood Method (SNM).
    Sorts S1 and target records along multiple sorting keys and sweeps a sliding
    window of width W across the sorted list.
    Complexity: O(N log N + N * W)
    """

    def __init__(self, window_size: int = 7):
        self.window_size = window_size

    @staticmethod
    def generate_sort_keys(record: dict) -> List[str]:
        name = (record.get('name_clean', '') or record.get('business_name', '')).lower()
        addr = (record.get('address_clean', '') or record.get('business_address', '')).lower()
        country = (record.get('country', '') or '').upper().strip()

        keys = []
        name_clean_compact = re.sub(r'[^a-z0-9]', '', name)
        addr_clean_compact = re.sub(r'[^a-z0-9]', '', addr)

        # Sort Key 1: Name prefix (first 8 chars) + Country
        if name_clean_compact:
            keys.append((f"K1_{name_clean_compact[:8]}_{country}", record['entity_id'], record))

        # Sort Key 2: Soundex of first token + Name prefix
        first_word = name.split()[0] if name.split() else ""
        if first_word:
            snd = generate_soundex_key(first_word)
            keys.append((f"K2_{snd}_{name_clean_compact[:6]}", record['entity_id'], record))

        # Sort Key 3: Reversed name prefix (groups entities with similar suffixes / trade names)
        if len(name_clean_compact) >= 4:
            rev_name = name_clean_compact[::-1]
            keys.append((f"K3_{rev_name[:8]}_{country}", record['entity_id'], record))

        # Sort Key 4: Address prefix + Country
        if addr_clean_compact:
            keys.append((f"K4_{addr_clean_compact[:8]}_{country}", record['entity_id'], record))

        return keys

    def block(self, s1_records: List[dict], target_records: List[dict]) -> Dict[str, Set[str]]:
        candidates = defaultdict(set)
        s1_ids = {r['entity_id'] for r in s1_records}

        # We evaluate across multiple sorting passes for maximum recall
        all_records = s1_records + target_records

        # Pass keys: K1, K2, K3, K4
        for pass_prefix in ["K1_", "K2_", "K3_", "K4_"]:
            keyed_records = []
            for rec in all_records:
                for k_tuple in self.generate_sort_keys(rec):
                    if k_tuple[0].startswith(pass_prefix):
                        keyed_records.append(k_tuple)

            # Sort lexicographically by key
            keyed_records.sort(key=lambda x: x[0])

            n = len(keyed_records)
            w = self.window_size

            for i in range(n):
                cur_key, cur_id, _ = keyed_records[i]
                if cur_id not in s1_ids:
                    continue

                # Look behind and ahead within window
                start = max(0, i - w)
                end = min(n, i + w + 1)
                for j in range(start, end):
                    if i == j:
                        continue
                    _, other_id, _ = keyed_records[j]
                    if other_id not in s1_ids:
                        candidates[cur_id].add(other_id)

        return candidates


# =====================================================================
# 4. MinHash & Locality-Sensitive Hashing (LSH)
# =====================================================================

class MinHashLSHBlocker:
    """
    MinHash & Locality Sensitive Hashing (LSH) Blocker.
    Partitions character 3-gram / word shingle MinHash signatures into b bands of r rows.
    Entities that collide in at least one band bucket are considered candidate matches.
    Pure-Python standard library implementation with zero external dependencies.
    """

    def __init__(self, num_perm: int = 64, num_bands: int = 16, threshold: float = 0.3):
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self.threshold = threshold

        # Universal hash function parameters: (a * x + b) % prime
        self.prime = 2147483647  # Mersenne prime (2^31 - 1)
        # Deterministic pseudo-random coefficients for reproducibility
        self.hash_params = []
        for i in range(num_perm):
            a = (i * 1000003 + 49999) % (self.prime - 1) + 1
            b = (i * 7919 + 6359) % self.prime
            self.hash_params.append((a, b))

        self.buckets = defaultdict(list)

    def _get_shingles(self, text: str) -> Set[int]:
        """Extracts character 3-gram and word shingles as 32-bit integer hashes."""
        shingles = set()
        clean = re.sub(r'\s+', ' ', text.lower().strip())
        if not clean:
            return shingles

        # Character 3-grams
        for i in range(max(1, len(clean) - 2)):
            tri = clean[i:i+3]
            h = int(hashlib.md5(tri.encode('utf-8')).hexdigest()[:8], 16)
            shingles.add(h)

        # Word shingles
        words = clean.split()
        for w in words:
            if len(w) >= 3:
                h = int(hashlib.md5(w.encode('utf-8')).hexdigest()[:8], 16)
                shingles.add(h)

        return shingles

    def _compute_minhash_signature(self, shingles: Set[int]) -> List[int]:
        """Computes MinHash signature vector of length num_perm."""
        if not shingles:
            return [0] * self.num_perm

        sig = []
        for a, b in self.hash_params:
            min_val = min(((a * s + b) % self.prime) for s in shingles)
            sig.append(min_val)
        return sig

    def fit(self, target_records: List[dict]):
        self.buckets.clear()
        for rec in target_records:
            t_id = rec['entity_id']
            text = f"{rec.get('name_clean', '')} {rec.get('address_clean', '')}"
            shingles = self._get_shingles(text)
            sig = self._compute_minhash_signature(shingles)

            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band_sub_sig = tuple(sig[start:end])
                bucket_key = (band_idx, band_sub_sig)
                self.buckets[bucket_key].append(t_id)

    def query(self, s1_records: List[dict]) -> Dict[str, Set[str]]:
        candidates = defaultdict(set)
        for rec in s1_records:
            s1_id = rec['entity_id']
            text = f"{rec.get('name_clean', '')} {rec.get('address_clean', '')}"
            shingles = self._get_shingles(text)
            sig = self._compute_minhash_signature(shingles)

            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band_sub_sig = tuple(sig[start:end])
                bucket_key = (band_idx, band_sub_sig)

                for t_id in self.buckets.get(bucket_key, []):
                    candidates[s1_id].add(t_id)

        return candidates


# =====================================================================
# 5. TF-IDF Cosine Similarity Top-K Blocker
# =====================================================================

class TfidfCosineBlocker:
    """
    TF-IDF Cosine Similarity Top-K Blocker.
    Uses scikit-learn when available; falls back to high-performance standard library
    inverted index TF-IDF when running in pure standard Python.
    """

    def __init__(self, top_k: int = 15, min_sim: float = 0.15):
        self.top_k = top_k
        self.min_sim = min_sim
        self.has_sklearn = False
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.neighbors import NearestNeighbors
            self.has_sklearn = True
        except ImportError:
            self.has_sklearn = False

    def block(self, s1_records: List[dict], target_records: List[dict]) -> Dict[str, Set[str]]:
        if not s1_records or not target_records:
            return defaultdict(set)

        if self.has_sklearn:
            return self._block_sklearn(s1_records, target_records)
        else:
            return self._block_pure_python(s1_records, target_records)

    def _block_sklearn(self, s1_records: List[dict], target_records: List[dict]) -> Dict[str, Set[str]]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors

        target_ids = [r['entity_id'] for r in target_records]
        target_texts = [f"{r.get('name_clean', '')} {r.get('address_clean', '')}".strip() for r in target_records]
        s1_texts = [f"{r.get('name_clean', '')} {r.get('address_clean', '')}".strip() for r in s1_records]

        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        target_matrix = tfidf.fit_transform(target_texts)
        s1_matrix = tfidf.transform(s1_texts)

        n_neighbors = min(self.top_k, target_matrix.shape[0])
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
        nn.fit(target_matrix)

        distances, indices = nn.kneighbors(s1_matrix)

        candidates = defaultdict(set)
        for s1_idx, neighbor_indices in enumerate(indices):
            s1_id = s1_records[s1_idx]['entity_id']
            for rank, n_idx in enumerate(neighbor_indices):
                cosine_sim = 1.0 - distances[s1_idx][rank]
                if cosine_sim >= self.min_sim or rank < 3:
                    cand_id = target_ids[n_idx]
                    candidates[s1_id].add(cand_id)

        return candidates

    def _block_pure_python(self, s1_records: List[dict], target_records: List[dict]) -> Dict[str, Set[str]]:
        """Pure-Python standard library sparse TF-IDF inverted index."""
        target_ids = [r['entity_id'] for r in target_records]

        # 1. Build document frequency across targets
        df = Counter()
        doc_terms = []
        for r in target_records:
            text = f"{r.get('name_clean', '')} {r.get('address_clean', '')}".lower()
            tokens = re.findall(r'[a-z0-9]{3,}', text)
            term_set = set(tokens)
            for t in term_set:
                df[t] += 1
            doc_terms.append(Counter(tokens))

        num_targets = len(target_records)
        idf = {t: math.log((num_targets + 1) / (count + 1)) + 1.0 for t, count in df.items()}

        # 2. Build inverted index with normalized TF-IDF weights
        inverted_index = defaultdict(list)
        for idx, (t_id, term_counts) in enumerate(zip(target_ids, doc_terms)):
            norm_sq = 0.0
            weighted_terms = {}
            for t, count in term_counts.items():
                w = (1.0 + math.log(count)) * idf.get(t, 1.0)
                weighted_terms[t] = w
                norm_sq += w * w
            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
            for t, w in weighted_terms.items():
                inverted_index[t].append((t_id, w / norm))

        # 3. Query S1 entities
        candidates = defaultdict(set)
        for r in s1_records:
            s1_id = r['entity_id']
            text = f"{r.get('name_clean', '')} {r.get('address_clean', '')}".lower()
            tokens = re.findall(r'[a-z0-9]{3,}', text)
            if not tokens:
                continue

            q_counts = Counter(tokens)
            q_norm_sq = 0.0
            q_weights = {}
            for t, count in q_counts.items():
                if t in idf:
                    w = (1.0 + math.log(count)) * idf[t]
                    q_weights[t] = w
                    q_norm_sq += w * w
            q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0

            scores = Counter()
            for t, w in q_weights.items():
                norm_w = w / q_norm
                for t_id, t_weight in inverted_index.get(t, []):
                    scores[t_id] += norm_w * t_weight

            top_matches = scores.most_common(self.top_k)
            for t_id, sim in top_matches:
                if sim >= self.min_sim or len(candidates[s1_id]) < 3:
                    candidates[s1_id].add(t_id)

        return candidates


# =====================================================================
# 6. Multi-Pass Union Blocker (Master Blocker)
# =====================================================================

class MultiPassBlocker:
    """
    Comprehensive Multi-Pass Union Blocker.
    Unions candidates from:
    - Pass 1: Informative Token Inverted Index (filtered for high-frequency terms)
    - Pass 2: Phonetic Inverted Index (Soundex & Metaphone)
    - Pass 3: Country-Aware Composite Blocking Keys
    - Pass 4: Sorted Neighborhood Method (SNM) with multi-key sliding window
    - Pass 5: MinHash & Locality-Sensitive Hashing (LSH)
    - Pass 6: TF-IDF Cosine Similarity Top-K Neighbors

    Guarantees:
    - Every S1 entity has a deterministic row in the candidate map.
    - Strictly only S2- and S3- target entity IDs.
    - Strictly NO S1- self-matches.
    - Deduplicated candidate lists.
    - Capped at max_candidates_per_entity to maintain high reduction ratio.
    """

    def __init__(
        self,
        top_k_tfidf: int = 15,
        min_ngram_overlap: int = 2,
        snm_window_size: int = 7,
        lsh_num_bands: int = 16,
        max_candidates_per_entity: int = 100,
        enable_lsh: bool = True,
        enable_snm: bool = True,
        enable_country_keys: bool = True,
        enable_tfidf: bool = True
    ):
        self.top_k_tfidf = top_k_tfidf
        self.min_ngram_overlap = min_ngram_overlap
        self.snm_window_size = snm_window_size
        self.lsh_num_bands = lsh_num_bands
        self.max_candidates_per_entity = max_candidates_per_entity

        self.enable_lsh = enable_lsh
        self.enable_snm = enable_snm
        self.enable_country_keys = enable_country_keys
        self.enable_tfidf = enable_tfidf

        self.country_blocker = CountryAwareBlocker()
        self.snm_blocker = SortedNeighborhoodBlocker(window_size=snm_window_size)
        self.lsh_blocker = MinHashLSHBlocker(num_bands=lsh_num_bands)
        self.tfidf_blocker = TfidfCosineBlocker(top_k=top_k_tfidf)

    def fit_transform(self, s1_records: List[dict], target_records: List[dict]) -> Dict[str, List[str]]:
        """
        Executes all active blocking passes, merges candidates, enforces challenge constraints,
        and returns mapping: s1_id -> sorted list of candidate target IDs.
        """
        if not s1_records:
            return {}

        valid_target_ids = {r['entity_id'] for r in target_records if not r['entity_id'].startswith("S1-")}
        candidate_scores = defaultdict(Counter)

        # -------------------------------------------------------------
        # Pass 1 & 2: Token Inverted Index + Phonetic Index
        # -------------------------------------------------------------
        token_index = defaultdict(list)
        phonetic_index = defaultdict(list)
        doc_freq = Counter()

        for rec in target_records:
            t_id = rec['entity_id']
            if t_id not in valid_target_ids:
                continue

            name = rec.get('name_clean', '') or rec.get('business_name', '')
            tokens = [w for w in name.lower().split() if len(w) >= 3]
            seen_tokens = set(tokens)

            for tok in seen_tokens:
                doc_freq[tok] += 1
                token_index[tok].append(t_id)

                snd = generate_soundex_key(tok)
                if snd:
                    phonetic_index[snd].append(t_id)

        # Prune ultra-frequent stopword tokens (> 20% of target records)
        max_df = max(10, int(len(target_records) * 0.20))
        pruned_token_index = {tok: ids for tok, ids in token_index.items() if len(ids) <= max_df}

        for s1_rec in s1_records:
            s1_id = s1_rec['entity_id']
            name = s1_rec.get('name_clean', '') or s1_rec.get('business_name', '')
            tokens = [w for w in name.lower().split() if len(w) >= 3]

            token_hits = Counter()
            for tok in set(tokens):
                for t_id in pruned_token_index.get(tok, []):
                    token_hits[t_id] += 2

                snd = generate_soundex_key(tok)
                for t_id in phonetic_index.get(snd, []):
                    token_hits[t_id] += 1

            for t_id, score in token_hits.items():
                if score >= self.min_ngram_overlap:
                    candidate_scores[s1_id][t_id] += score

        # -------------------------------------------------------------
        # Pass 3: Country-Aware Composite Blocking Keys
        # -------------------------------------------------------------
        if self.enable_country_keys:
            self.country_blocker.fit(target_records)
            country_cands = self.country_blocker.query(s1_records)
            for s1_id, t_ids in country_cands.items():
                for t_id in t_ids:
                    if t_id in valid_target_ids:
                        candidate_scores[s1_id][t_id] += 3

        # -------------------------------------------------------------
        # Pass 4: Sorted Neighborhood Method (SNM)
        # -------------------------------------------------------------
        if self.enable_snm:
            snm_cands = self.snm_blocker.block(s1_records, target_records)
            for s1_id, t_ids in snm_cands.items():
                for t_id in t_ids:
                    if t_id in valid_target_ids:
                        candidate_scores[s1_id][t_id] += 2

        # -------------------------------------------------------------
        # Pass 5: MinHash & Locality-Sensitive Hashing (LSH)
        # -------------------------------------------------------------
        if self.enable_lsh:
            self.lsh_blocker.fit(target_records)
            lsh_cands = self.lsh_blocker.query(s1_records)
            for s1_id, t_ids in lsh_cands.items():
                for t_id in t_ids:
                    if t_id in valid_target_ids:
                        candidate_scores[s1_id][t_id] += 3

        # -------------------------------------------------------------
        # Pass 6: TF-IDF Cosine Top-K Neighbors
        # -------------------------------------------------------------
        if self.enable_tfidf:
            tfidf_cands = self.tfidf_blocker.block(s1_records, target_records)
            for s1_id, t_ids in tfidf_cands.items():
                for t_id in t_ids:
                    if t_id in valid_target_ids:
                        candidate_scores[s1_id][t_id] += 4

        # -------------------------------------------------------------
        # Format final candidate map per S1 entity
        # -------------------------------------------------------------
        final_candidate_map = {}
        for s1_rec in s1_records:
            s1_id = s1_rec['entity_id']
            scored_candidates = candidate_scores.get(s1_id, Counter())

            # Sort by accumulated multi-pass score descending, then alphabetical
            sorted_candidates = [
                cand_id for cand_id, _ in sorted(
                    scored_candidates.items(),
                    key=lambda item: (-item[1], item[0])
                )
            ]

            # Filter strictly: valid target IDs only, no S1 self-matches
            filtered = [cid for cid in sorted_candidates if cid in valid_target_ids and not cid.startswith("S1-")]

            # Cap max candidates per S1 entity
            if self.max_candidates_per_entity and len(filtered) > self.max_candidates_per_entity:
                filtered = filtered[:self.max_candidates_per_entity]

            final_candidate_map[s1_id] = sorted(filtered)

        return final_candidate_map


# =====================================================================
# 7. Evaluation & Quality Metrics for Blocking
# =====================================================================

def compute_blocking_metrics(
    candidate_map: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    total_target_count: int
) -> Dict[str, Any]:
    """
    Computes key performance metrics for blocking stage:
    1. Recall Ceiling: Fraction of true matching pairs captured in the candidate set.
    2. Reduction Ratio: 1 - (candidates generated / total Cartesian product pairs).
    3. Pair Completeness and Candidate Statistics.
    """
    total_true_pairs = 0
    retained_true_pairs = 0
    total_candidates_generated = 0
    candidate_lengths = []
    zero_candidate_count = 0

    s1_entities_in_gt = set(ground_truth.keys())

    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        total_true_pairs += len(true_matches)
        generated_cands = set(candidate_map.get(s1_id, []))
        retained = len(set(true_matches) & generated_cands)
        retained_true_pairs += retained

    for s1_id, cands in candidate_map.items():
        c_len = len(cands)
        candidate_lengths.append(c_len)
        total_candidates_generated += c_len
        if c_len == 0:
            zero_candidate_count += 1

    num_s1 = max(len(candidate_map), 1)
    recall_ceiling = retained_true_pairs / max(total_true_pairs, 1)
    total_possible_pairs = num_s1 * max(total_target_count, 1)
    reduction_ratio = 1.0 - (total_candidates_generated / max(total_possible_pairs, 1))

    avg_cands = total_candidates_generated / num_s1
    median_cands = (
        sorted(candidate_lengths)[len(candidate_lengths) // 2]
        if candidate_lengths else 0
    )

    return {
        "recall_ceiling": recall_ceiling,
        "reduction_ratio": reduction_ratio,
        "total_true_pairs": total_true_pairs,
        "retained_true_pairs": retained_true_pairs,
        "missed_true_pairs": total_true_pairs - retained_true_pairs,
        "total_candidates": total_candidates_generated,
        "avg_candidates_per_entity": avg_cands,
        "median_candidates": median_cands,
        "max_candidates": max(candidate_lengths) if candidate_lengths else 0,
        "min_candidates": min(candidate_lengths) if candidate_lengths else 0,
        "zero_candidate_entities": zero_candidate_count,
        "total_s1_entities": num_s1
    }
