from seqeval.metrics import precision_score as seqeval_precision
from seqeval.metrics import recall_score as seqeval_recall
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import classification_report as seqeval_classification_report

from jiwer import cer

from commons import decompose_into_punct_tags

def generate_bio_tags(aligned_tokens):
    """
    Generates BIO tags for aligned tokens with NER tags.
    Args:
        aligned_tokens (list): Aligned tokens (including None for unaligned).
    Returns:
        bio_tags (list): BIO tags corresponding to the tokens.
    """
    bio_tags = []
    current_entity = None
    current_opening=None

    for token in aligned_tokens:
        if token in ["=", "*"]:
            bio_tags.append("O")  # Unaligned tokens are "Outside"
            continue

        # Check for opening and closing NER tags
        otag, mid, ctag, _ = decompose_into_punct_tags(token)
        if otag:
            current_opening = otag[1:-1]  # Extract NER type (e.g., "DATE")

        if mid:
          if current_entity:
            # Inside an entity: B-<ENTITY> for the first token, I-<ENTITY> for the rest
            if len(bio_tags) == 0 or bio_tags[-1] == "O":
                bio_tags.append(f"B-{current_entity}")
                current_opening=None
            else:
                bio_tags.append(f"I-{current_entity}")
          else:
              if current_opening:
                current_entity = current_opening
                bio_tags.append(f"B-{current_entity}")
                current_opening=None
              else:
                bio_tags.append("O")

        # Check for a closing tag in the token
        if ctag:
            current_entity = None  # End the current entity
            current_opening=None

    return bio_tags


def calculate_ner_success(true_labels, predicted_labels):
    """
    Calculates NER success metrics using the seqeval library.

    Args:
        true_labels (list of list of str): The ground truth labels for the dataset.
                                           Each inner list represents the labels for a single sentence.
        predicted_labels (list of list of str): The predicted labels for the dataset.
                                                Each inner list represents the labels for a single sentence.

    Returns:
        dict: A dictionary containing precision, recall, F1-score, and a classification report.
    """
    # Validate input
    if len(true_labels) != len(predicted_labels):
        raise ValueError("The number of true and predicted label sequences must be the same.")

    # Calculate metrics
    precision = seqeval_precision(true_labels, predicted_labels)
    recall = seqeval_recall(true_labels, predicted_labels)
    f1 = seqeval_f1(true_labels, predicted_labels)
    report = seqeval_classification_report(true_labels, predicted_labels)

    # Return metrics as a dictionary
    return precision, recall, f1, report
    

def calculate_ner_success_on_aligned(aligned_ref_tokens, aligned_hyp_tokens):
  ref_bio_tags = generate_bio_tags(aligned_ref_tokens)
  hyp_bio_tags = generate_bio_tags(aligned_hyp_tokens)

  precision, recall, f1, report = calculate_ner_success([ref_bio_tags], [hyp_bio_tags])
  return {"ref_bio_tags": ref_bio_tags, "hyp_bio_tags": hyp_bio_tags, "precision": precision, "recall": recall, "f1": f1, "report": report}

'''
t1 = [["O", "O", "B-MONEY", "O", "B-PERSON"]]
t2 = [["O", "O", "B-MONEY", "O", "B-CARD"]]

p, r, f, rep = calculate_ner_success(t1, t2)
print(p, r, f)
print(rep)
'''
