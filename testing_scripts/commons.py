import re, string

punct_marks = (".", "!", ";", ",", "?")

opening_tags = open("opening_tags.txt", "r").read().split("\n")
opening_tags = [ot for ot in opening_tags if ot and ot.strip()]
closing_tags = open("closing_tags.txt", "r").read().split("\n")
closing_tags = [ct for ct in closing_tags if ct and ct.strip()]

def is_token_punct(token):
  return token in string.punctuation

def decompose_into_punct_tags(token):
  otag, mid, ctag, punct = "", token, "", ""
  for tag in opening_tags:
    if token.startswith(tag):
      otag = tag
      token = token[len(tag):]
      break

  for punctm in punct_marks:
    if token.endswith(punctm):
      token = token[:-1]
      punct = punctm
      break

  for tag in closing_tags:
    if token.endswith(tag):
      ctag = tag
      token = token[:-len(tag)]
      break

  return otag, token, ctag, punct


def strip_puncts_tags(token):
  _, mid, _, _ = decompose_into_punct_tags(token)
  return mid

def strip_tags(token):
  _, mid, _, punct = decompose_into_punct_tags(token)
  return mid+punct

def find_punct(token):
  pot_p = token[-1]
  if pot_p in punct_marks:
    return pot_p
  return None

#===============
def normalize_ent(ent_text):
  newtoks = []
  ent_tokens = ent_text.strip().split()
  for token in ent_tokens:
    token = strip_tags(token)
    if token in ["#", "*"]:
      pass
    else:
      newtoks.append(token)
  return " ".join(newtoks)


#==============

def preprocess_text_for_alignment(text):
    """
    Preprocesses text for alignment by lowercasing all words
    but preserving NER tags and punctuation.

    Args:
        text (str): Input text.

    Returns:
        str: Preprocessed text.
    """
    def lowercase_token(token):
      if token.startswith("[") or token[-1] == "]" or token[:-2].endswith("]"):
        otag, mid, ctag, punct = decompose_into_punct_tags(token)
        token = otag + mid.lower() + ctag + punct
      else:
        token = token.lower()
      return token

    tokens = text.split()
    preprocessed_tokens = [lowercase_token(token) for token in tokens]

    # Reconstruct the text
    return " ".join(preprocessed_tokens)



#==============
# Runtime clean

def token_has_two_closing_tags(token):
  return "][/" in token

def token_has_two_opening_tags(token):
  return "][" in token and not token_has_two_closing_tags(token)

def has_many_tags(token):
  return "][" in token

def fix_two_opening_tags(token):
  extra_tag, mid, c_tag, punct = decompose_into_punct_tags(token)
  return mid+c_tag+punct

def fix_two_closing_tags(token):
  o_tag, mid, extra_tag, punct = decompose_into_punct_tags(token)
  return o_tag+mid+punct

def fix_too_many_tags(token):
  if token_has_two_closing_tags(token):
    token = fix_two_closing_tags(token)

  if token_has_two_opening_tags(token):
    token = fix_two_opening_tags(token)
  return token

def is_token_single_tag(token):
  if token in opening_tags or token in closing_tags:
    return True
  extra_tag, mid, c_tag, punct = decompose_into_punct_tags(token)
  if not mid or not mid.strip():
    return True
  return False


def clean_non_ascii_punct(text):
    # Define a whitelist of common ASCII punctuation to keep
    whitelist = string.punctuation  # Includes !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

    # Match any character that is a punctuation but not in the whitelist
    text = re.sub(r'[^\w\s' + re.escape(whitelist) + r']+', '', text)
    return text

def preclean(text):
  new_tokens = []
  text = text.replace("*|", " ")
  del_tokens = ['"', "'", "&", "’", "‘", "”", "“", "—", "(", ")", "{", "}"]
  for delt in del_tokens:
    text = text.replace(delt, " ")
  text = re.sub(r"(\d+) %", r"\1%" , text)
  text = re.sub(r"(\[[A-Z_]+\]) ([A-Za-z0-9])", r"\1\2", text)
  text = re.sub(r"([A-Za-z0-9]) (\[/[A-Z_]+\])", r"\1\2", text)
  text = clean_non_ascii_punct(text)
  tokens = text.strip().split()
  for token in tokens:
    if is_token_single_tag(token):
      pass
    elif has_many_tags(token):
      token = fix_too_many_tags(token)
      if is_token_single_tag(token):
        pass
      else:
        new_tokens.append(token)
    elif token in string.punctuation: # kill dangling marks
      pass
    else:
      new_tokens.append(token)
  return " ".join(new_tokens)

def clean_punct_mistakes(text):
  text = re.sub(r"\].(\w)", r"]. \1", text)
  text = re.sub(r"(\w).(\w)", r"\1 \2", text)
  text = re.sub(r"([A-Za-z]),([A-Za-z])", r"\1, \2", text)
  return text


def correct_tags(text):
    tokens = text.strip().split()
    newtokens = []
    all_tags = open("wrong_tags.txt", "r").read().split("\n")
    all_tags = [at for at in all_tags if at and at.strip()]
    for token in tokens:
      for tag in all_tags:
          if token.startswith(tag):
              taglen = len(tag)
              tagtext = tag[1:-1]
              #print(tagtext, "tagtext")
              #print()
              token = token[taglen:]
              #print(token, "token after slice")
              token = "[" + tagtext + "]" + token
              #print(token, "new token")
              #print("-------------------")
          elif tag in token:
              taglen = len(tag)
              if "/" in tag:
                tagtext = tag[2:-1]
                newtagtext = "[/" + tagtext + "]"
              else:
                tagtext = tag[1:-1]
                newtagtext = "[" + tagtext + "]"

              token = token.replace(tag, newtagtext)
      newtokens.append(token)
    newt = " ".join(newtokens)
    return newt


#==========================================

def normalize_text(text):
    puncts = string.punctuation
    tokens = text.strip().split()
    newtoks = []
    for token in tokens:
      if not token or token in puncts:
        pass
      token = strip_puncts_tags(token)
      if token:
        newtoks.append(token)
    return " ".join(newtoks)


#================


def handle_mismatched_tags(text, timeout=5):
  tokens = text.split()
  newtokens = []
  current_entity=None
  for i, token in enumerate(tokens):
    kill=False
    otag, mid, ctag, punct = decompose_into_punct_tags(token)
    if otag:
      current_entity =otag[1:-1] 

      if ctag:
        ent = ctag[2:-1]
        if ent == current_entity:
          current_entity=None
          newtokens.append(token)
        else:
          # tag mismatch kill both tags
          token = mid+punct
          newtokens.append(token)
        continue
      
      next_tokens = tokens[i+1:i+timeout+1]
      #print(token, current_entity, next_tokens)
      if not next_tokens:
        kill=True
      for ntoken in next_tokens:
        notag, nmid, nctag, npunct = decompose_into_punct_tags(ntoken)
        #print(ntoken, notag, nctag)
        if notag:
          kill = True
          current_entity = None
          break
        elif nctag:
          ent = nctag[2:-1]
          if ent == current_entity:
            break # good, closing found
          else:
            # dangling close
            kill=True
            current_entity = None
            break
    elif ctag:
      if not current_entity:
        kill = True
    else:
      pass

    if kill:
      token = mid+punct

    newtokens.append(token)
  return " ".join(newtokens)
