from ner_metrics import generate_bio_tags
from extract_entity_text import extract_entity_texts
'''
tokens1 = ["[PERSON]Bob", "Dylan[/PERSON]", "born", "[DATE]May", "24,", "1941[/DATE]"]
generate_bio_tags(tokens1)
tokens1 = ["[PERSON]Bob", "Dylan[/PERSON]", "born", "as", "[PERSON]Robert", "Allen", "Zimmerman[/PERSON]", "[DATE]May", "24,", "1941[/DATE]"]
x = generate_bio_tags(tokens1)



from commons import handle_mismatched_tags

#x = handle_mismatched_tags(tokens1)

tokens2 = ["[PERSON]Allen", "Poe", "Zimmerman[/PERSON]", "[DATE]1923", "May", "[DATE]4"]
#print(tokens2)
#x = handle_mismatched_tags(tokens2)
#print(x)

tokens2 = ["[PERSON]Allen", "Poe", "Zimmerman[/PERSON]", "[DATE]1923", "May", "[DATE]4[/DATE]"]
print(tokens2)
x = handle_mismatched_tags(tokens2)
print(x)
'''

reference_tokens = ["I", "spent", "$100.000", ".", "Bryan", "Adams", "="]
hypothesis_tokens = ["I", "spent", "$100000", ".", "Bryan", "Adams", "Getty"]

# Corresponding BIO tags
reference_bio = ["O", "O", "B-MONEY", "O", "B-PERSON", "I-PERSON", "O"]
hypothesis_bio = ["O", "O", "B-MONEY", "O", "B-PERSON", "I-PERSON", "I-PERSON" ]

# Call the function
entity_mapping = extract_entity_texts(reference_tokens, hypothesis_tokens, reference_bio, hypothesis_bio)

# Output
for ref_entity, hyp_entity, typ in entity_mapping:
    print(f"{ref_entity} -> {hyp_entity}  {typ}")


reference_tokens = ["I", "spent", "$100.000", ".", "Bryan", "Adams"]
hypothesis_tokens = ["I", "spent", "$100000", ".", "Bryan", "Adams"]

# Corresponding BIO tags
reference_bio = ["O", "O", "B-MONEY", "O", "B-PERSON", "I-PERSON"]
hypothesis_bio = ["O", "O", "B-MONEY", "O", "B-PERSON", "I-PERSON"]

# Call the function
entity_mapping = extract_entity_texts(reference_tokens, hypothesis_tokens, reference_bio, hypothesis_bio)

# Output
for ref_entity, hyp_entity, typ in entity_mapping:
    print(f"{ref_entity} -> {hyp_entity}  {typ}")



reference_tokens = ["I", "spent", "$100.000", ".", "Bryan", "Adams"]
hypothesis_tokens = ["I", "spent", "$100000", ".", "Bryan", "*"]

# Corresponding BIO tags
reference_bio = ["O", "O", "B-MONEY", "O", "B-PERSON", "I-PERSON"]
hypothesis_bio = ["O", "O", "B-MONEY", "O", "B-PERSON", "O"]

entity_mapping = extract_entity_texts(reference_tokens, hypothesis_tokens, reference_bio, hypothesis_bio)
# Call the function
for ref_entity, hyp_entity, typ in entity_mapping:
    print(f"{ref_entity} -> {hyp_entity}  {typ}")

reference_tokens = ["I", "spent", "$100.000", ".", "Bryan", "Adams"]
hypothesis_tokens = ["I", "spent", "$100000", ".", "*", "*"]

# Corresponding BIO tags
reference_bio = ["O", "O", "B-MONEY", "O", "B-PERSON", "I-PERSON"]
hypothesis_bio = ["O", "O", "B-MONEY", "O", "O", "O"]

entity_mapping = extract_entity_texts(reference_tokens, hypothesis_tokens, reference_bio, hypothesis_bio)
# Call the function
for ref_entity, hyp_entity, typ in entity_mapping:
    print(f"{ref_entity} -> {hyp_entity}  {typ}")
