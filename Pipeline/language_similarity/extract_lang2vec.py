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
        Syntactic features compiled from sources like WALS, SSWL, and Ethnologue.
        Missing values are predicted using K-Nearest Neighbor (KNN) smoothing based on typological similarity
        to other languages. This preserves natural linguistic diversity.

    - 'syntax_average':
        Features are averaged across available sources (e.g., WALS + SSWL) when overlap exists.
        No prediction or imputation is done — missing values remain missing if not present in any source.

2. Phonology-based Features (sound system properties):
    - 'phonology_knn':
        Phonological properties (e.g., tone, syllable structure) based on WALS and Ethnologue.
        Missing values are predicted using KNN smoothing based on phonological similarity.

    - 'phonology_average':
        Feature values are averaged across available sources when multiple databases report them.
        Provides a unified representation without artificial prediction.

3. Phonological Inventory Features (detailed sound inventories):
    - 'inventory_knn':
        Binary phoneme presence vectors from PHOIBLE and similar databases.
        Missing entries are predicted using KNN smoothing from phonologically similar languages.

    - 'inventory_average':
        Phoneme presence values are averaged across overlapping sources, resulting in fractional presence indicators.
        Reflects aggregated observations from different phonological inventories.

Special Handling:
-----------------
- 'average' feature sets aggregate values across multiple typological databases without
    attempting to fill in missing entries. Missing values remain as-is.

- 'knn' feature sets use a 10-nearest neighbors classification method to impute missing values,
    leveraging a weighted combination of genetic, geographic, and structural distances between languages.
    This method achieves 92.93% accuracy in 10-fold cross-validation for predicting typological features.
    (https://aclanthology.org/E17-2002.pdf)
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

def save_results(results, feature_set):
    output_dir = f'../data/language_similarity/lang2vec/{feature_set}/'
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
    ]

    # Compute cosine similarities
    for feature_set in feature_sets:
        print(f"Computing similarities using {feature_set}...")
        similarities = compute_cosine_similarities(languages, feature_set)
        if similarities:
            save_results(similarities, feature_set)

if __name__ == "__main__":
    main()
