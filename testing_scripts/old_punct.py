from commons import decompose_into_punct_tags, is_token_punct, strip_tags, strip_puncts_tags, find_punct

def calculate_punct_cap_success(aligned_ref_tokens, aligned_hyp_tokens):
    """
    Calculates success rates for capitalization, punctuation, and NER tags using seqeval.

    Args:
        reference (str): Reference text.
        hypothesis (str): Hypothesis text (transcription).

    Returns:
        dict: Dictionary containing success rates and NER tag evaluation.
    """

    # Initialize counters
    capitalization_matches = 0
    capitalization_total = 0

    punctuation_matches = {}
    punctuation_totals = {}

    # Regular expressions
    punctuation_regex = r"[.,!?;:]"

    # Compare aligned tokens
    for ref_token, hyp_token in zip(aligned_ref_tokens, aligned_hyp_tokens):
        if ref_token == "#" or hyp_token == "*":  # Skip unaligned tokens
            continue

        ref_token_puncted, hyp_token_puncted = strip_tags(ref_token), strip_tags(hyp_token)
        ref_token_raw, hyp_token_raw = strip_puncts_tags(ref_token), strip_puncts_tags(hyp_token)

        # Capitalization: Compare lowercased version of tokens
        if ref_token_raw.lower() == hyp_token_raw.lower():
            capitalization_total += 1
            if ref_token_raw == hyp_token_raw:  # Exact match includes capitalization
                capitalization_matches += 1

        # Punctuation: Check for punctuation marks in tokens
        ref_punct = find_punct(ref_token)
        hyp_punct = find_punct(hyp_token)

        if ref_punct:
            if ref_punct not in punctuation_totals:
                punctuation_totals[ref_punct] = 0
                punctuation_matches[ref_punct] = 0

            punctuation_totals[ref_punct] += 1
            if ref_punct == hyp_punct:
                punctuation_matches[ref_punct] += 1

    # Calculate success rates
    capitalization_success = capitalization_matches / capitalization_total if capitalization_total > 0 else 0
    punctuation_success = {
        punct: punctuation_matches[punct] / punctuation_totals[punct]
        for punct in punctuation_totals
    }

    return {"punctuation_totals": punctuation_totals, "punctuation_matches": punctuation_matches,  "cap_totals": capitalization_total, capitalization_matches)}



'''
ref = "I went there before [DATE]Q1[/DATE] and we saved a total of <MONEY>$1.000.000</MONEY>."
hypo = "You went there before [CARDINAL]Q1[/CARDINAL] and we saved a total of <MONEY>$1.000.000</MONEY>."


ref = preprocess_text_for_alignment(ref)
hypo = preprocess_text_for_alignment(hypo)
print(ref)
print(hypo)

result = align_texts(ref, hypo)
aref, ahyp, ops = result

print("===========================")

print(" ".join(aref))
print(" ".join(ahyp))
print(" ".join(ops))

'''
