def extract_entity_texts(reference_tokens, hypothesis_tokens, reference_bio, hypothesis_bio, lookahead=5):
    """
    Extract the entity text for each entity in the reference and map it to the hypothesis,
    including the entity type from the reference. Uses lookahead to ensure the full hypothesis entity is captured.

    Args:
        reference_tokens (list): Tokens in the reference (aligned).
        hypothesis_tokens (list): Tokens in the hypothesis (aligned).
        reference_bio (list): BIO tags for the reference tokens.
        hypothesis_bio (list): BIO tags for the hypothesis tokens.
        lookahead (int): Number of tokens to look ahead in the hypothesis when collecting entities.

    Returns:
        entity_mapping (list): A list of tuples containing:
            (reference_entity_text, hypothesis_entity_text, reference_entity_type)
    """
    entity_mapping = []
    current_ref_entity = []
    current_entity_type = None

    i = 0
    while i < len(reference_tokens):
        ref_token = reference_tokens[i]
        hyp_token = hypothesis_tokens[i]
        ref_bio = reference_bio[i]
        hyp_bio = hypothesis_bio[i]

        # Check if the current reference token starts a new entity
        if ref_bio.startswith("B-"):
            # Finalize any previous entity
            if current_ref_entity:
                reference_entity_text = " ".join(current_ref_entity)
                hypothesis_entity_text = " ".join(current_hyp_entity) if current_hyp_entity else ""
                entity_mapping.append((reference_entity_text, hypothesis_entity_text, current_entity_type))
            
            # Start a new entity
            current_entity_type = ref_bio[2:]  # Extract the entity type (e.g., "MONEY")
            current_ref_entity = [ref_token] if ref_token not in ["=", "*"] else []

            # Collect the hypothesis entity using lookahead
            current_hyp_entity = []
            if hyp_bio.startswith("B-"):
                j = i
                while j < len(hypothesis_tokens) and j < i + lookahead:
                    if hypothesis_bio[j].startswith("B-") or hypothesis_bio[j].startswith("I-"):
                        current_hyp_entity.append(hypothesis_tokens[j])
                    else:
                        break
                    j += 1

        # If the current reference token is inside an entity
        elif ref_bio.startswith("I-") and ref_bio[2:] == current_entity_type:
            if ref_token not in ["=", "*"]:
                current_ref_entity.append(ref_token)

        # If the entity ends in the reference
        elif ref_bio == "O" and current_ref_entity:
            # Finalize the current entity
            reference_entity_text = " ".join(current_ref_entity)
            hypothesis_entity_text = " ".join(current_hyp_entity) if current_hyp_entity else ""
            entity_mapping.append((reference_entity_text, hypothesis_entity_text, current_entity_type))
            
            # Reset
            current_ref_entity = []
            current_hyp_entity = []
            current_entity_type = None

        i += 1

    # Handle any remaining entity at the end of the tokens
    if current_ref_entity:
        reference_entity_text = " ".join(current_ref_entity)
        hypothesis_entity_text = " ".join(current_hyp_entity) if current_hyp_entity else ""
        entity_mapping.append((reference_entity_text, hypothesis_entity_text, current_entity_type))
    
    return entity_mapping
