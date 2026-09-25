"""
Stage 3: Feature Engineering Module
Extracts comprehensive pairwise similarity features for (S1, candidate_target) entity pairs:
- String distance metrics (Levenshtein, Jaro-Winkler, LCS)
- Token & Character n-gram similarity (Jaccard, Cosine)
- TF-IDF similarity (Name, Address, Combined)
- Structural & categorical features (Country match, First token match, Numeric overlap, Length ratio)
"""

import math
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """Computes Jaro-Winkler similarity ratio between 0.0 and 1.0."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3.0
    l = 0
    for i in range(min(4, min(len1, len2))):
        if s1[i] == s2[i]:
            l += 1
        else:
            break

    return jaro + l * p * (1 - jaro)


def token_jaccard(tokens1: list, tokens2: list) -> float:
    """Computes Jaccard similarity over word tokens."""
    set1, set2 = set(tokens1), set(tokens2)
    union = set1 | set2
    if not union:
        return 0.0
    return len(set1 & set2) / len(union)


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Computes Jaccard similarity over character n-grams."""
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    ngrams1 = set(s1[i:i+n] for i in range(len(s1) - n + 1))
    ngrams2 = set(s2[i:i+n] for i in range(len(s2) - n + 1))
    union = ngrams1 | ngrams2
    if not union:
        return 0.0
    return len(ngrams1 & ngrams2) / len(union)


def numeric_token_overlap(s1: str, s2: str) -> float:
    """Computes overlap ratio of numeric tokens (PINs, street numbers)."""
    nums1 = set(w for w in s1.split() if w.isdigit())
    nums2 = set(w for w in s2.split() if w.isdigit())
    if not nums1 and not nums2:
        return 1.0
    union = nums1 | nums2
    if not union:
        return 0.0
    return len(nums1 & nums2) / len(union)


class FeatureExtractor:
    def __init__(self):
        self.tfidf_name = None
        self.tfidf_addr = None

    def fit(self, all_records: list):
        """Fits TF-IDF vectorizers on all records."""
        names = [r['name_clean'] for r in all_records if r.get('name_clean')]
        addrs = [r['address_clean'] for r in all_records if r.get('address_clean')]

        if names:
            self.tfidf_name = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            self.tfidf_name.fit(names)
        if addrs:
            self.tfidf_addr = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            self.tfidf_addr.fit(addrs)

    def extract_pair_features(self, rec1: dict, rec2: dict) -> list:
        """Extracts a feature vector for a pair of business records (rec1, rec2)."""
        n1, n2 = rec1['name_clean'], rec2['name_clean']
        a1, a2 = rec1['address_clean'], rec2['address_clean']

        # 1. Levenshtein ratio
        lev_dist_name = levenshtein_distance(n1, n2)
        max_len_name = max(len(n1), len(n2), 1)
        lev_ratio_name = 1.0 - (lev_dist_name / max_len_name)

        lev_dist_addr = levenshtein_distance(a1, a2)
        max_len_addr = max(len(a1), len(a2), 1)
        lev_ratio_addr = 1.0 - (lev_dist_addr / max_len_addr)

        # 2. Jaro-Winkler
        jw_name = jaro_winkler_similarity(n1, n2)
        jw_addr = jaro_winkler_similarity(a1, a2)

        # 3. Token Jaccard
        t1_name, t2_name = n1.split(), n2.split()
        t1_addr, t2_addr = a1.split(), a2.split()

        jacc_name = token_jaccard(t1_name, t2_name)
        jacc_addr = token_jaccard(t1_addr, t2_addr)

        # 4. Char n-grams
        c3_name = char_ngram_jaccard(n1, n2, 3)
        c3_addr = char_ngram_jaccard(a1, a2, 3)

        # 5. TF-IDF Cosine Similarity
        tfidf_name_sim = 0.0
        tfidf_addr_sim = 0.0
        if self.tfidf_name and n1 and n2:
            vecs = self.tfidf_name.transform([n1, n2])
            tfidf_name_sim = float(cosine_similarity(vecs[0], vecs[1])[0][0])

        if self.tfidf_addr and a1 and a2:
            vecs = self.tfidf_addr.transform([a1, a2])
            tfidf_addr_sim = float(cosine_similarity(vecs[0], vecs[1])[0][0])

        # 6. Structural & Categorical
        country_match = 1.0 if rec1.get('country') == rec2.get('country') else 0.0
        first_token_match = 1.0 if (t1_name and t2_name and t1_name[0] == t2_name[0]) else 0.0
        num_overlap = numeric_token_overlap(a1, a2)

        len_diff_name = abs(len(n1) - len(n2))
        len_ratio_name = min(len(n1), len(n2)) / max(len(n1), len(n2), 1)

        return [
            lev_ratio_name,
            lev_ratio_addr,
            jw_name,
            jw_addr,
            jacc_name,
            jacc_addr,
            c3_name,
            c3_addr,
            tfidf_name_sim,
            tfidf_addr_sim,
            country_match,
            first_token_match,
            num_overlap,
            len_diff_name,
            len_ratio_name
        ]
