# Business Entity Resolution Methodology Write-Up

## Team & Overview
- **Challenge**: Amazon ML Challenge 2026 - Business Entity Resolution
- **Target Metric**: Macro F_0.5 Score (Precision weighted 2× over Recall)
- **License Compliance**: MIT / Apache 2.0 open-source models (< 8 Billion parameters)
- **Data Policy**: Zero external database/API lookups (fully self-contained)

---

## 1. Methodology Overview
Our solution adopts a 4-stage modular architecture designed for high scalability, strong precision tuning, and strict compliance with competition constraints:

1. **Preprocessing & Normalization**: Standardizes noisy text, extracts canonical legal suffixes, expands street abbreviations, protects landmark phrases as informative signals, and normalizes open-set country strings without restrictive filtering.
2. **Multi-Pass Fuzzy Blocking**: Combines token-level inverted indexes, phonetic sound-alike keying, first-token + location keys, and TF-IDF Cosine Nearest-Neighbors to achieve > 98% recall ceiling while drastically reducing candidate pairs.
3. **Pairwise Feature Engineering & GBDT Classifier**: Computes 15+ similarity features across edit distance, token overlap, character n-grams, TF-IDF cosine, and structural numeric attributes. Trains a LightGBM / Gradient Boosting classifier to predict entity alignment probabilities.
4. **Macro F_0.5 Threshold Calibration & Singleton Management**: Calibrates the acceptance decision threshold on a held-out validation split to maximize macro F_0.5. Strictly handles singletons (entities with zero true matches) by emitting empty candidate lists.

---

## 2. Candidate Generation & Blocking Strategy (Lead: Person 2)
Because testing every Source 1 entity against all Source 2 and Source 3 entities yields an intractable Cartesian product search space ($|S_1| \times (|S_2| + |S_3|)$) and injects massive negative noise, the candidate generation (blocking) stage determines the upper bound on the entire pipeline's recall. Person 2 is responsible for designing, benchmarking, and executing this multi-pass candidate generation system.

### Multi-Pass Blocking Architecture
To guarantee that true business entity matches are not discarded while eliminating over 90–99% of negative pairs, we deploy a 6-pass complementary union blocking architecture:

1. **Informative Token Inverted Index**:
   - Indexes word tokens of length $\ge 3$.
   - Applies frequency pruning: terms appearing in $> 20\%$ of target records (e.g., generic industry stopwords like "enterprises", "solutions") are filtered out to prevent Cartesian explosions in common buckets.
2. **Phonetic Key Indexing (Soundex & Simplified Metaphone)**:
   - Maps tokens into American Soundex (4-character alphanumeric phoneme) and Metaphone representations.
   - Bridges phonetic transcription gaps, transliteration spelling divergence, and acoustic spelling errors (e.g., "Philip" $\leftrightarrow$ "Filip", "Centre" $\leftrightarrow$ "Center").
3. **Country-Aware Composite Blocking Keys**:
   - Combines primary entity tokens and numeric address identifiers with country labels:
     - `FW_{token}_{country}`: Primary business name token partitioned by country.
     - `SND_{soundex}_{country}`: Phonetic Soundex code partitioned by country.
     - `PAIR_{token1}_{token2}_{country}`: First two sorted tokens partitioned by country.
     - `NUM_{number}_{country}`: Address numeric tokens (PIN codes, street numbers) partitioned by country.
     - `CROSS_COUNTRY_FALLBACK`: Country-agnostic token keying to handle missing or open-set country annotations (e.g., France, India, US).
4. **Sorted Neighborhood Method (SNM)**:
   - Sorts all records lexicographically across multiple keys: (a) Normalized name prefix (first 8 chars), (b) Soundex + name prefix, (c) Reversed name prefix (to group entities with similar suffixes), and (d) Address prefix.
   - Slides a window of size $W = 7$ across the sorted array, linking records within the sliding neighborhood in $O(N \log N + N \cdot W)$ time.
5. **MinHash & Locality-Sensitive Hashing (LSH)**:
   - Extracts character 3-gram and token shingles from normalized text.
   - Computes $H = 64$ MinHash signatures using universal hash functions $h_i(x) = (a_i \cdot x + b_i) \pmod p$.
   - Divides signatures into $b = 16$ bands of $r = 4$ rows. Entities colliding in at least one band bucket are linked as candidate pairs with theoretical collision probability $1 - (1 - s^r)^b$, capturing high Jaccard similarity pairs sub-linearly.
6. **TF-IDF Cosine Similarity Top-$K$ Retrieval**:
   - Computes sublinear TF-IDF vectors over character (range 2–4) and word (range 1–2) n-grams.
   - Identifies top $K = 20$ nearest target neighbors per Source 1 entity via cosine similarity.

### Evaluation Metrics & Validation Results
We measure blocking quality using two primary metrics on a 20% held-out validation split:

$$\text{Recall Ceiling} = \frac{|\mathcal{M}^* \cap \mathcal{C}|}{|\mathcal{M}^*|}, \quad \text{Reduction Ratio} = 1 - \frac{|\mathcal{C}|}{|S_1| \times (|S_2| + |S_3|)}$$

where $\mathcal{M}^*$ is the set of ground-truth true matching pairs, and $\mathcal{C}$ is the candidate pair set.

| Blocking Strategy | Recall Ceiling | Reduction Ratio | Avg Candidates / S1 |
| :--- | :---: | :---: | :---: |
| 1. Inverted Token Index | 100.0% | 93.21% | 21.1 |
| 2. Country-Aware Composite Keys | 100.0% | 94.22% | 17.9 |
| 3. Sorted Neighborhood (SNM) | 28.12% | 97.36% | 8.2 |
| 4. MinHash / LSH (Bands=16) | 97.50% | 94.48% | 17.1 |
| 5. TF-IDF Cosine Top-15 | 81.25% | 95.39% | 14.3 |
| **6. Full Multi-Pass Ensemble** | **100.00%** | **90.00%** | **31.0** |

All candidate outputs strictly adhere to competition constraints: tab-separated format, exactly one row per Source 1 entity, only valid S2/S3 entity IDs (zero self-matches to S1), and zero duplicates per list.

---

## 3. Model Architecture & Feature Engineering

### Feature Matrix (15 Pairwise Features)
1. **Levenshtein Similarity Ratio (Name & Address)**: Edit distance normalized by string length.
2. **Jaro-Winkler Distance (Name & Address)**: Gives higher weights to matching prefix strings.
3. **Token Jaccard Similarity (Name & Address)**: Intersection-over-union of word token sets.
4. **Char 3-Gram Jaccard Similarity (Name & Address)**: Captures subtle spelling variations and typos.
5. **TF-IDF Cosine Similarity (Name & Address)**: Measures term importance similarity.
6. **Country Exact Match**: Binary flag (1 if open-set country strings match, 0 otherwise).
7. **First Token Match**: Binary flag indicating exact match on leading entity token.
8. **Numeric Token Overlap Ratio**: Match ratio of numbers (street/house numbers, PIN codes).
9. **Length Ratio & Absolute Length Difference**: Structural size disparity features.

### Classifier Architecture
- **Model**: LightGBM Classifier / Gradient Boosting Decision Trees
- **Parameters**: `n_estimators=150`, `learning_rate=0.05`, `max_depth=5`, `num_leaves=31`
- **License**: MIT License (Compliant with < 8B parameter rule)

---

## 4. Evaluation, Singleton & Open-Set Country Strategy

### Macro F_0.5 Calibration
The F_0.5 metric heavily penalizes false positives (false merges):
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
We tune the acceptance probability threshold $t^*$ on our validation split specifically for Macro F_0.5 (yielding an optimal threshold around 0.55–0.65). This conservative threshold prevents merging distinct businesses.

### Singleton Handling
Entities without true matches (singletons) score 1.0 when correctly left empty (`""`), and 0.0 if any false match is predicted. Our high precision threshold ensures singletons are accurately identified and left unmerged.

### Open-Set Country Generalization
Training data contains records from US and India, while test data additionally introduces France. All normalization, blocking, and feature extraction components are rule-free from fixed country lookups. Country features rely on generic equality string matching, guaranteeing seamless generalization to unseen countries like France.
