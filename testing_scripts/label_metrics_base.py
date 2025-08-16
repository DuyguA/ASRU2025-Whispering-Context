import jellyfish
from jiwer import cer
from commons import decompose_into_punct_tags
from collections import defaultdict
from commons import normalize_ent
from jellyfish import jaro_winkler_similarity

def extract_entity_texts(reference_tokens, hypothesis_tokens, reference_bio):
    """
    Extract the entity text for each entity in the reference and map it to the hypothesis,
    including the entity type from the reference.

    Args:
        reference_tokens (list): Tokens in the reference (aligned).
        hypothesis_tokens (list): Tokens in the hypothesis (aligned).
        reference_bio (list): BIO tags for the reference tokens.

    Returns:
        entity_mapping (list): A list of tuples containing:
            (reference_entity_text, hypothesis_entity_text, reference_entity_type)
    """
    entity_mapping = []
    current_ref_entity = []
    current_hyp_entity = []
    current_entity_type = None

    for i in range(len(reference_tokens)):
        ref_token = reference_tokens[i]
        hyp_token = hypothesis_tokens[i]
        ref_bio = reference_bio[i]

        # Check if the current reference token starts a new entity
        if ref_bio.startswith("B-"):
            # Finalize any previous entity
            if current_ref_entity:
                reference_entity_text = " ".join(current_ref_entity)
                hypothesis_entity_text = " ".join(current_hyp_entity)
                entity_mapping.append((reference_entity_text, hypothesis_entity_text, current_entity_type))

            # Start a new entity
            current_entity_type = ref_bio[2:]  # Extract the entity type (e.g., "MONEY")
            current_ref_entity = [ref_token]
            current_hyp_entity = [hyp_token]

        # If the current token is part of the same entity
        elif ref_bio.startswith("I-") and current_entity_type:
            current_ref_entity.append(ref_token)
            current_hyp_entity.append(hyp_token)

        # If the current token is outside any entity
        else:
            # Finalize any previous entity
            if current_ref_entity:
                reference_entity_text = " ".join(current_ref_entity)
                hypothesis_entity_text = " ".join(current_hyp_entity)
                entity_mapping.append((reference_entity_text, hypothesis_entity_text, current_entity_type))

            # Reset for the next entity
            current_ref_entity = []
            current_hyp_entity = []
            current_entity_type = None

    # Finalize the last entity if any
    if current_ref_entity:
        reference_entity_text = " ".join(current_ref_entity)
        hypothesis_entity_text = " ".join(current_hyp_entity)
        entity_mapping.append((reference_entity_text, hypothesis_entity_text, current_entity_type))

    return entity_mapping


def calculate_jaro_winkler(entities):
    """
    Calculate the average Jaro-Winkler similarity for a list of (reference, hypothesis) pairs.
    """
    # Calculate Jaro-Winkler similarity for each pair
    jaro_winkler_scores = [
        jaro_winkler_similarity(ref_text, hyp_text) if hyp_text else 0
        for ref_text, hyp_text in entities
    ]

    # Avoid division by zero if entities list is empty
    return sum(jaro_winkler_scores) / len(jaro_winkler_scores) if jaro_winkler_scores else 0.0

def calculate_cer(pairs):
    cer_scores = [cer(ref, hyp) for ref, hyp in pairs]  # Calculate CER for each pair

    # Calculate average CER, avoid division by zero if pairs is empty
    return sum(cer_scores) / len(cer_scores) if cer_scores else 0.0

jaro_class = [
"DATE",
"EVENT",
"FAC",
"GPE",
"LANGUAGE",
"LAW",
"LOC",
"NORP",
"ORDINAL",
"ORG",
"PERSON",
"PRODUCT",

]

cer_class = [
"CARDINAL",
"NUMERIC",
"MONEY",
"PERCENT",
"QUANTITY",
"TIME",
"URL",
"EMAIL",
]



def calculate_label_stats(reference_tokens, hypothesis_tokens, reference_bio):
  entity_mapping = defaultdict(list)
  entity_success_jw = {}
  entity_success_cer = {}

  entity_triplets = extract_entity_texts(reference_tokens, hypothesis_tokens, reference_bio)
  for ref_ent, hyp_ent, ent_label in entity_triplets:
    ref_ent = normalize_ent(ref_ent)
    hyp_ent = normalize_ent(hyp_ent)
    entity_mapping[ent_label].append((ref_ent, hyp_ent))

  for label, ent_pairs in entity_mapping.items():
    if label in jaro_class:
      js = calculate_jaro_winkler(ent_pairs)
      entity_success_jw[label] = js
    else:
      cs =  calculate_cer(ent_pairs)
      entity_success_cer[label] = cs
  return {"ent_mapping": entity_mapping, "jw_dict": entity_success_jw, "cer_dict": entity_success_cer}

def calculate_all_label_stats(entity_mapping):
  entity_success_jw = {}
  entity_success_cer = {}


  for label, ent_pairs in entity_mapping.items():
    if label in jaro_class:
      js = calculate_jaro_winkler(ent_pairs)
      entity_success_jw[label] = js
    else:
      cs =  calculate_cer(ent_pairs)
      entity_success_cer[label] = cs
  return {"jw_dict": entity_success_jw, "cer_dict": entity_success_cer}

