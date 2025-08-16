from jiwer import process_words


def align_texts(ref_text, hyp_text):
    """
    Aligns reference and hypothesis tokens using JIWER.

    Args:
        ref_text (str): Reference text.
        hyp_text (str): Hypothesis text.

    Returns:
        tuple: Aligned reference and hypothesis tokens.
    """
    # Process words using JIWER
    alignment = process_words(ref_text, hyp_text)

    # Extract references, hypotheses, and alignment chunks
    ref_tokens = alignment.references[0]
    hyp_tokens = alignment.hypotheses[0]
    align_chunks = alignment.alignments[0]

    # Create aligned reference and hypothesis tokens
    aligned_ref_tokens = []
    aligned_hyp_tokens = []

    for chunk in align_chunks:
        if chunk.type == "equal":  # Matching tokens
            aligned_ref_tokens.extend(ref_tokens[chunk.ref_start_idx:chunk.ref_end_idx])
            aligned_hyp_tokens.extend(hyp_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx])
        elif chunk.type == "substitute":  # Substituted tokens
            aligned_ref_tokens.extend(ref_tokens[chunk.ref_start_idx:chunk.ref_end_idx])
            aligned_hyp_tokens.extend(hyp_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx])
        elif chunk.type == "delete":  # Missing token in hypothesis
            aligned_ref_tokens.extend(ref_tokens[chunk.ref_start_idx:chunk.ref_end_idx])
            aligned_hyp_tokens.extend(["*"] * (chunk.ref_end_idx - chunk.ref_start_idx))  # Placeholder for missing hypothesis word
        elif chunk.type == "insert":  # Extra token in hypothesis
            aligned_ref_tokens.extend(["#"] * (chunk.hyp_end_idx - chunk.hyp_start_idx))  # Placeholder for missing reference word
            aligned_hyp_tokens.extend(hyp_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx])

    return aligned_ref_tokens, aligned_hyp_tokens

