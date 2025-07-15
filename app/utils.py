
from rapidfuzz import process, fuzz


def normalize_keys_fuzzy(input_dict, mapping, threshold=80):
    normalized = {}

   
    all_synonyms = []
    synonym_to_key = {}
    for key, synonyms in mapping.items():
        for syn in synonyms:
            all_synonyms.append(syn.lower())
            synonym_to_key[syn.lower()] = key

    for k, v in input_dict.items():
        k_lower = k.lower()
        # البحث عن أفضل تطابق
        best_match, score, _ = process.extractOne(k_lower, all_synonyms, scorer=fuzz.ratio)
        if score >= threshold:
            normalized_key = synonym_to_key[best_match]
            normalized[normalized_key] = v
        else:
            # لو لم يجد تطابق جيد، خلي المفتاح كما هو
            normalized[k] = v

    return normalized


