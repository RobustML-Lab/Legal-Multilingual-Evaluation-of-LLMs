"""
===========================
Language Feature Sets Used
===========================

This script computes language similarities based on multiple feature sets extracted from the lang2vec dataset.
Each feature set represents a different linguistic aspect of languages, allowing us to later correlate linguistic
distance with LLM performance on multilingual tasks.

Feature Set Categories:
------------------------

1. Syntax-based Features (grammar structure):
    - 'syntax_knn':
        Features extracted from WALS, Ethnologue, and SSWL databases.
        Missing values are filled using K-Nearest Neighbor (KNN) smoothing based on similar languages.
        This maintains real-world linguistic diversity.

    - 'syntax_average':
        Features from the same sources but missing values are filled by simple averaging.
        This can artificially reduce diversity but provides a simple baseline for comparison.

2. Phonology-based Features (sound systems):
    - 'phonology_knn':
        Phonological properties (like tones, vowel/consonant inventories) using KNN smoothing.
        Captures sound similarity while respecting cross-linguistic variation.

    - 'phonology_average':
        Similar features averaged across available sources.
        Useful for simple comparison but may smooth over important phonetic distinctions.

3. Phonological Inventory Features (detailed sound inventories):
    - 'inventory_knn':
        Full phonological inventories (lists of sounds) filled with KNN smoothing.
        Important for studying languages at the level of their phonetic details (phones, tones).

    - 'inventory_average':
        Average of phonological inventories from multiple phonetic databases.
        Smoother but less faithful to linguistic differences.

4. Learned Embeddings:
    - 'learned':
        Dense embeddings learned from training data (e.g., multilingual corpora).
        Captures abstract language properties not tied directly to linguistics (good for experiments but harder to interpret).

5. Categorical Metadata:
    - 'fam':
        Language family information.
        This is NOT numeric. It is categorical (same family = 1, different family = 0).
        Used for analyzing if languages from the same family perform similarly.

6. Geographic Metadata:
    - 'geo':
        Geographic coordinates (latitude, longitude) of languages.
        Used to calculate geographic distance (Euclidean distance in lat-long space).
        Helpful for analyzing if physical proximity between languages affects model performance.

Special Handling:
-----------------
- For feature sets like 'syntax_average', 'phonology_average', missing values ('--') are replaced with 0.0
  during vector cleaning to ensure numerical stability for cosine similarity computation.

- 'fam' and 'geo' are NOT cosine similarities:
    - 'fam' is binary: 1 (same family) or 0 (different families).
    - 'geo' computes Euclidean distance (geographic distance).

"""


from lang2vec.lang2vec import get_features
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os

def compute_cosine_similarities(languages, feature_set):
    try:
        features = get_features(languages, feature_set)
    except Exception as e:
        print(f"Error fetching features for {feature_set}: {e}")
        return []

    language_names = list(features.keys())
    vectors = []

    for lang in language_names:
        raw_vector = features[lang]
        # Replace '--' with 0.0
        clean_vector = [float(x) if x != '--' else 0.0 for x in raw_vector]
        vectors.append(clean_vector)

    results = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            lang1, lang2 = language_names[i], language_names[j]
            sim = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
            results.append({
                'language_1': lang1,
                'language_2': lang2,
                'similarity': sim,
                'feature_set': feature_set
            })
    return results


def compute_geo_distances(languages):
    try:
        features = get_features(languages, 'geo')
    except Exception as e:
        print(f"Error fetching geo features: {e}")
        return []

    language_names = list(features.keys())
    coords = list(features.values())

    results = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            lang1, lang2 = language_names[i], language_names[j]
            dist = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            results.append({
                'language_1': lang1,
                'language_2': lang2,
                'distance': dist,
                'feature_set': 'geo'
            })
    return results

def compute_family_match(languages):
    try:
        features = get_features(languages, 'fam')
    except Exception as e:
        print(f"Error fetching family features: {e}")
        return []

    language_names = list(features.keys())
    families = list(features.values())

    results = []
    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            lang1, lang2 = language_names[i], language_names[j]
            same_family = 1 if families[i] == families[j] else 0
            results.append({
                'language_1': lang1,
                'language_2': lang2,
                'same_family': same_family,
                'feature_set': 'fam'
            })
    return results

def save_results(results, feature_set):
    output_dir = f'../output/language_similarity/lang2vec/{feature_set}/'
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    filename = 'language_similarity.csv' if feature_set != 'geo' and feature_set != 'fam' else 'language_distance.csv'
    df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"Saved {feature_set} similarities/distances to {output_dir}")

def main():
    languages = [
        'eng', 'fra', 'deu', 'spa', 'ita', 'nld', 'por', 'pol', 'swe', 'dan', 'nor',
        'fin', 'ell', 'ron', 'mlt', 'hun', 'ces', 'slk', 'bul', 'hrv', 'lit', 'lav', 'est', 'isl',
        'zho', 'ara', 'tur', 'hin', 'ben', 'jpn', 'kor', 'rus', 'urd', 'vie', 'tha', 'ind', 'fas'
    ]

    # Numeric feature sets for cosine similarity
    feature_sets = [
        'syntax_knn', 'syntax_average',
        'phonology_knn', 'phonology_average',
        'inventory_knn', 'inventory_average',
        'learned'
    ]

    # Compute cosine similarities
    for feature_set in feature_sets:
        print(f"Computing similarities using {feature_set}...")
        similarities = compute_cosine_similarities(languages, feature_set)
        if similarities:
            save_results(similarities, feature_set)

    # Geo (compute distance)
    print("Computing geographic distances...")
    geo_distances = compute_geo_distances(languages)
    if geo_distances:
        save_results(geo_distances, 'geo')

    # Family match (same or different)
    print("Computing family matches...")
    family_matches = compute_family_match(languages)
    if family_matches:
        save_results(family_matches, 'fam')

if __name__ == "__main__":
    main()
