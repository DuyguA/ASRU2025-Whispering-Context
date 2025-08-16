from jiwer import wer, cer, compute_measures
from commons import normalize_text

def calculate_wer(reference, hypothesis):
    """
    Calculates the Word Error Rate (WER) for normalized text using jiwer.

    Args:
        reference (str): The ground truth/reference text.
        hypothesis (str): The predicted text.

    Returns:
        dict: A dictionary containing WER, CER, and detailed measures (substitutions, insertions, deletions).
    """
    # Define a text normalization pipeline

    # Normalize reference and hypothesis
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)

    # Compute WER and character error rate (optional)
    wer_value = wer(reference, hypothesis)
    cer_value = cer(reference, hypothesis)

    # Compute detailed measures (substitutions, deletions, insertions)
    details = compute_measures(reference, hypothesis)
    matches, inserts, subs, dels = details["hits"], details["insertions"], details["substitutions"], details["deletions"]

    return {
        "WER": wer_value,
        "CER": cer_value,
        "Matches": matches,
        "Insertions": inserts,
        "Substitutions": subs,
        "Deletions": dels,
    }


#print(calculate_wer("You go there.", "You went there yaaa!"))
