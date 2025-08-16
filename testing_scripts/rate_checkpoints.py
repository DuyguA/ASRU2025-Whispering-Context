from make_alignment import align_texts
from normalized_wer import calculate_wer
from punct_cap_metrics import calculate_punct_cap_success
from ner_metrics import calculate_ner_success_on_aligned

from commons import preclean, correct_tags, clean_punct_mistakes, handle_mismatched_tags
from collections import defaultdict
from entity_metrics import calculate_per_entity_stats


def find_deletion_segment_start(hypo, threshold=30):
    # Initialize counters
    count = 0
    start_index = -1

    # Iterate through the sequence
    for i, char in enumerate(hypo):
        if char == '*':
            if count == 0:
                start_index = i  # Mark the start of the segment
            count += 1
            # If the consecutive '*' count reaches the threshold, return the start index
            if count >= threshold:
                return start_index
        else:
            # Reset the count when a non-'*' character is encountered
            count = 0
            start_index = -1

    # If no segment is found, return -1
    return -1


def remake_text(aligned_ref_tokens, aligned_hyp_tokens):
  new_ref, new_hypo = [], []
  for token1, token2 in zip(aligned_ref_tokens, aligned_hyp_tokens):
      if token1 == "=" and token2 == "*":
        pass
      else:
        new_ref.append(token1)
        new_hypo.append(token2)
  return " ".join(new_ref), " ".join(new_hypo)


def process_single_pair(reference, hypothesis):
  reference = correct_tags(reference)
  #hypothesis = clean_punct_mistakes(hypothesis)
  reference, hypothesis = preclean(reference), preclean(hypothesis)
  hypothesis = handle_mismatched_tags(hypothesis)

  aligned_ref_tokens, aligned_hyp_tokens = align_texts(reference, hypothesis)
  start_stars = find_deletion_segment_start(aligned_hyp_tokens)
  if start_stars > 0:
    aligned_ref_tokens, aligned_hyp_tokens = aligned_ref_tokens[:start_stars], aligned_hyp_tokens[:start_stars]
    reference, hypothesis = remake_text(aligned_ref_tokens, aligned_hyp_tokens)
  ner_tag_results  = calculate_ner_success_on_aligned(aligned_ref_tokens, aligned_hyp_tokens)
  label_stats = calculate_per_entity_stats(aligned_ref_tokens, aligned_hyp_tokens, ner_tag_results["ref_bio_tags"], ner_tag_results["hyp_bio_tags"])
  normalized_results = calculate_wer(reference, hypothesis)
  return normalized_results, ner_tag_results, label_stats
  


def process_single_test_instance(base_name, ref_text, hypo_text):
  wer_results= process_single_pair(ref_text, hypo_text)
  return wer_results


chkpnts = [
"4",
"9",
"14",
"19",
"24",
"29",
"34",
"39",
"44",
"49",
"54",
"59",
"64",
"69",
"74",
"79",
"84",
"89",
"94",
"99",
]

chkpnts = ["99"]
  
def process_all(model_type):
  success_file =  "reports/whisper_" + model_type + "_checkpoints" + ".txt"
  base_name = "Bob_Dylan"
  ref_dir = "../test_results/ground_truth/"
  hypo_dir = "../checkpoint_eval/all_puncted/"

  ref_file = ref_dir + base_name + ".txt"
  ref_text = open(ref_file, "r").read().strip()

  with open(success_file, "w") as osfile:
    for chkpnt in chkpnts:
        hypo_file = hypo_dir + chkpnt + ".txt"

        hypo_text = open(hypo_file, "r").read().strip()
        print(chkpnt)

        wer_results, ner_results, label_stats = process_single_test_instance(base_name, ref_text, hypo_text)
        file_wer, file_cer, matches, inserts, subs, dels = wer_results.values()
        total_tokens = matches + subs + dels

        ref_bio_tags, hyp_bio_tags, precision_ner, recall_ner, f1_ner, report_ner = ner_results.values()
        print(label_stats)


        osfile.write(f"Filename: {chkpnt}\n")
        osfile.write(f"Normalized WER Stats:\n")
        osfile.write(f"WER: {file_wer:.2f}\n" )
        osfile.write(f"CER: {file_cer:.2f}\n" )
        osfile.write(f"Reference length: {total_tokens}\n" )
        osfile.write(f"Matches: {matches}\n" )
        osfile.write(f"Insertions: {inserts}\n" )
        osfile.write(f"Substitutions: {subs}\n" )
        osfile.write(f"Deletions: {dels}\n" )
        osfile.write("\n")
        osfile.write("\n")
        osfile.write(f"NER Stats:\n")
        osfile.write(f"F1: {f1_ner:.2f}\n" )
        osfile.write(f"Precision: {precision_ner:.2f}\n" )
        osfile.write(f"Recall: {recall_ner:.2f}\n" )
        osfile.write(f"Classification Report\n" )
        osfile.write(report_ner)
        osfile.write("\n")
        osfile.write("\n")
        osfile.write("==================================")
        osfile.write("\n")
        osfile.write("\n")
  

process_all("puncted")
