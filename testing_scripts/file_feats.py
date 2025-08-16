from make_alignment import align_texts
from normalized_wer import calculate_wer
from punct_cap_metrics import calculate_punct_cap_success, cap_success, punct_success
from ner_metrics import calculate_ner_success_on_aligned, calculate_ner_success
from label_metrics import calculate_label_stats, calculate_all_label_stats

from commons import preclean, correct_tags, clean_punct_mistakes, handle_mismatched_tags
from collections import defaultdict


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
  reference, hypothesis = preclean(reference), preclean(hypothesis)
  hypothesis = handle_mismatched_tags(hypothesis)

  aligned_ref_tokens, aligned_hyp_tokens = align_texts(reference, hypothesis)
  start_stars = find_deletion_segment_start(aligned_hyp_tokens)
  if start_stars > 0:
    aligned_ref_tokens, aligned_hyp_tokens = aligned_ref_tokens[:start_stars], aligned_hyp_tokens[:start_stars]
    reference, hypothesis = remake_text(aligned_ref_tokens, aligned_hyp_tokens)
  ner_tag_results  = calculate_ner_success_on_aligned(aligned_ref_tokens, aligned_hyp_tokens)
  label_stats = calculate_label_stats(aligned_ref_tokens, aligned_hyp_tokens, ner_tag_results["ref_bio_tags"], ner_tag_results["hyp_bio_tags"])
  punct_cap_results = calculate_punct_cap_success(aligned_ref_tokens, aligned_hyp_tokens)
  normalized_results = calculate_wer(reference, hypothesis)
  return normalized_results, punct_cap_results, ner_tag_results, label_stats


  
def process_single_test_instance(base_name, ref_text, hypo_text):
  wer_results, punct_cap_results, ner_results, label_stats = process_single_pair(ref_text, hypo_text)
  #print(wer_results)
  #print(punct_cap_results)
  #print(ner_results)
  return wer_results, punct_cap_results, ner_results, label_stats

  
def process_all_test_instances(test_file_names, model_type):
  success_file =  "reports/whisper_" + model_type + ".txt"
  ref_dir = "../test_results/ground_truth/"
  hypo_dir = "../test_results/whisper-" + model_type + "/"

  all_matches, all_inserts, all_subs, all_dels = 0, 0, 0, 0 
  all_ref_ner_tags = []
  all_hyp_ner_tags = []

  all_caps_refs = []
  all_caps_preds =[]

  all_punct_refs = defaultdict(list)
  all_punct_preds = defaultdict(list)

  all_label_mapping = defaultdict(list)

  with open(success_file, "w") as osfile:
    for base_name in test_file_names:
      ref_file = ref_dir + base_name + ".txt"
      hypo_file = hypo_dir + base_name + ".txt"

      ref_text = open(ref_file, "r").read().strip()
      hypo_text = open(hypo_file, "r").read().strip()

      wer_results, punct_cap_results, ner_results, label_stats = process_single_test_instance(base_name, ref_text, hypo_text)
      file_wer, file_cer, matches, inserts, subs, dels = wer_results.values()
      total_tokens = matches + dels + subs
      all_matches += matches
      all_inserts += inserts
      all_subs += subs
      all_dels +=dels

      capital_stats, punct_stats = punct_cap_results.values()
      true_labels_cap, pred_labels_cap,  precision_cap, recall_cap, f1_cap, acc_cap = capital_stats.values()
      num_cap =sum(true_labels_cap)
      num_pred_cap =sum(pred_labels_cap)

      all_caps_refs += true_labels_cap
      all_caps_preds += pred_labels_cap

      for punct_mark, succ_dict in punct_stats.items():
        ref = succ_dict["true_labels"]
        pred = succ_dict["pred_labels"]
        all_punct_refs[punct_mark] += ref
        all_punct_preds[punct_mark] += pred

      ref_bio_tags, hyp_bio_tags, precision_ner, recall_ner, f1_ner, report_ner = ner_results.values()
      all_ref_ner_tags.append(ref_bio_tags)
      all_hyp_ner_tags.append(hyp_bio_tags)

      label_mapping, jw_dict, cer_dict = label_stats.values()
      jw_dict = dict(sorted(jw_dict.items()))
      cer_dict = dict(sorted(cer_dict.items()))
      for label, ents_list in label_mapping.items():
          all_label_mapping[label] += ents_list


      osfile.write(f"Filename: {base_name}\n")
      osfile.write(f"Normalized WER Stats:\n")
      osfile.write(f"WER: {file_wer:.2f}\n" )
      osfile.write(f"CER: {file_cer:.2f}\n" )
      osfile.write(f"Reference length: {total_tokens}\n" )
      osfile.write(f"Matches: {matches}\n" )
      osfile.write(f"Insertions: {inserts}\n" )
      osfile.write(f"Substitutions: {subs}\n" )
      osfile.write(f"Deletions: {dels}\n" )
      osfile.write("----------------------\n")
      osfile.write(f"Capitalization Stats:\n")
      osfile.write(f"Reference number of capitals: {num_cap}\n" )
      osfile.write(f"Predicted number of capitals: {num_pred_cap}\n" )
      osfile.write(f"Accuracy: {acc_cap:.2f}\n" )
      osfile.write(f"F1: {f1_cap:.2f}\n" )
      osfile.write(f"Precision: {precision_cap:.2f}\n" )
      osfile.write(f"Recall: {recall_cap:.2f}\n" )
      osfile.write("----------------------\n")
      osfile.write(f"Punctuation Stats:\n")
      for punct_mark, succ_dict in punct_stats.items():
        refs, preds,  precision, recall, f1, acc = succ_dict.values()
        num_refs = sum(refs)
        num_preds = sum(preds)
        osfile.write(f"   Punct Mark: {punct_mark}\n" )
        osfile.write(f"   Reference Count: {num_refs}\n" )
        osfile.write(f"   Predictions Count: {num_preds}\n" )
        osfile.write(f"   Accuracy: {acc:.2f}\n" )
        osfile.write(f"   F1: {f1:.2f}\n" )
        osfile.write(f"   Precision: {precision:.2f}\n" )
        osfile.write(f"   Recall: {recall:.2f}\n" )
        osfile.write("    ----------------------\n")
      osfile.write("----------------------\n")
      osfile.write(f"NER Stats:\n")
      osfile.write(f"F1: {f1_ner:.2f}\n" )
      osfile.write(f"Precision: {precision_ner:.2f}\n" )
      osfile.write(f"Recall: {recall_ner:.2f}\n" )
      osfile.write(f"Classification Report\n" )
      osfile.write(report_ner)
      osfile.write("----------------------\n")
      osfile.write(f"Per label stats:\n")
      osfile.write(f"Labels evaluated by Jaro-Winkler score\n" )
      for label, score in jw_dict.items():
          osfile.write(f"   {label} {score:.2f}\n" )
      osfile.write("----------------------\n")
      osfile.write(f"Labels evaluated by CER\n" )
      for label, score in cer_dict.items():
          osfile.write(f"   {label} {score:.2f}\n" )
      osfile.write("----------------------\n")
      osfile.write("\n")
      osfile.write("\n")
      osfile.write("==================================")
      osfile.write("\n")
      osfile.write("\n")

    
  
  total_toks = all_matches + all_dels + all_subs
  all_wer = (all_subs + all_dels + all_inserts) / total_toks

  total_ner_prec, total_ner_recall, total_ner_f1, total_report = calculate_ner_success(all_ref_ner_tags, all_hyp_ner_tags)
  total_true_labels_cap, total_pred_labels_cap, total_precision_cap, total_recall_cap, total_f1_cap, total_acc_cap = cap_success(all_caps_refs, all_caps_preds).values()
  total_num_cap =sum(total_true_labels_cap)
  total_num_pred_cap =sum(total_pred_labels_cap)

 
  total_punct_stats = punct_success(all_punct_refs, all_punct_preds)
  total_jw_dict, total_cer_dict = calculate_all_label_stats(all_label_mapping).values()

  with open(success_file, "a+") as osfile:
    osfile.write("\n\n")
    osfile.write("ALL TEST FILES SUMMARY\n\n")
    osfile.write(f"Normalized WER Stats:\n")
    osfile.write(f"Total length: {total_toks}\n" )
    osfile.write(f"Total WER: {all_wer:.2f}\n" )
    osfile.write(f"Total matches: {all_matches}\n" )
    osfile.write(f"Total insertions: {all_inserts}\n" )
    osfile.write(f"Total substitutions: {all_subs}\n" )
    osfile.write(f"Total deletions: {all_dels}\n" )
    osfile.write("----------------------\n")
    osfile.write(f"Capitalization Stats:\n")
    osfile.write(f"Reference number of capitals: {total_num_cap}\n" )
    osfile.write(f"Predicted number of capitals: {total_num_pred_cap}\n" )
    osfile.write(f"Accuracy: {total_acc_cap:.2f}\n" )
    osfile.write(f"F1: {total_f1_cap:.2f}\n" )
    osfile.write(f"Precision: {total_precision_cap:.2f}\n" )
    osfile.write(f"Recall: {total_recall_cap:.2f}\n" )
    osfile.write("----------------------\n")
    osfile.write(f"Punctuation Stats:\n")
    for punct_mark, succ_dict in total_punct_stats.items():
        refs, preds,  precision, recall, f1, acc = succ_dict.values()
        num_refs = sum(refs)
        num_preds = sum(preds)
        osfile.write(f"   Punct Mark: {punct_mark}\n" )
        osfile.write(f"   Total reference count: {num_refs}\n" )
        osfile.write(f"   Total predictions Count: {num_preds}\n" )
        osfile.write(f"   Accuracy: {acc:.2f}\n" )
        osfile.write(f"   F1: {f1:.2f}\n" )
        osfile.write(f"   Precision: {precision:.2f}\n" )
        osfile.write(f"   Recall: {recall:.2f}\n" )
        osfile.write("    ----------------------\n")
    osfile.write("----------------------\n")
    osfile.write(f"NER Stats:\n")
    osfile.write(f"F1: {total_ner_f1:.2f}\n" )
    osfile.write(f"Precision: {total_ner_prec:.2f}\n" )
    osfile.write(f"Recall: {total_ner_recall:.2f}\n" )
    osfile.write(f"Classification Report\n" )
    osfile.write(total_report)
    osfile.write("----------------------\n")
    osfile.write(f"Per label stats:\n")
    osfile.write(f"Labels evaluated by Jaro-Winkler score\n" )
    for label, score in total_jw_dict.items():
      osfile.write(f"   {label} {score:.2f}\n" )
    osfile.write("----------------------\n")
    osfile.write(f"Labels evaluated by CER\n" )
    for label, score in total_cer_dict.items():
      osfile.write(f"   {label} {score:.2f}\n" )
    osfile.write("----------------------\n")
    osfile.write("\n")
    osfile.write("\n")






#process_single_test_instance("Bob_Dylan", "../test_results/ground_truth/", "../test_results/whisper-puncted/")

all_test_files = open("test_files.txt", "r").read().split("\n")
all_test_files = [atf for atf in all_test_files if atf and atf.strip()]

process_all_test_instances(all_test_files, "puncted")
