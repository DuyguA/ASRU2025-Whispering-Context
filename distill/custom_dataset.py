import torch
from torch.utils.data import Dataset, DataLoader
import json, os
from transformers import WhisperTokenizerFast
import math

def round_down_to_nearest_02(time_in_seconds):
    # Round down to the nearest 0.02 increment
    rounded_time = math.floor(time_in_seconds / 0.02) * 0.02
    return round(rounded_time, 2)




class PunctedDataset(Dataset):
  def __init__(self, phase="train", window_size=64):  # train test
    self.audio_path = "/ephemeral/featurize/feats/"
    self.chunk_file = "../text_chunks/segmented_puncted_" + phase + "_chunks.jsonl"
    self.chunks = []
    with open(self.chunk_file, "r") as injs:
      for line in injs:
        chunk = json.loads(line)
        self.chunks.append(chunk)

    self.tokenizer_path = "BayanDuygu/whisper-puncted-timed"
    self.tokenizer = WhisperTokenizerFast.from_pretrained(self.tokenizer_path)

    self.llama_dir = "/ephemeral/make_tensors/llama_tensors/"
    context_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    self.window_index = context_sizes.index(window_size)

  def __len__(self):
    return 2000

  def __getitem__(self, index):

    chunk = self.chunks[index]

    transcript = chunk["text"]
    start_time = chunk["start_time"]
    end_time = chunk["end_time"]
    m_stime = chunk["mstart_time"]
    r_stime = chunk["rstart_time"]
    chunk_id = chunk["chunk_id"]
    features_path = self.audio_path + "chunk" +  chunk_id + ".pt"

    if not os.path.isfile(features_path) or transcript.count("|") != 2:
        index = index // 3000
        chunk = self.chunks[index]
        transcript = chunk["text"]
        start_time = chunk["start_time"]
        end_time = chunk["end_time"]
        m_stime = chunk["mstart_time"]
        r_stime = chunk["rstart_time"]
        chunk_id = chunk["chunk_id"]
        features_path = self.audio_path + "chunk" +  chunk_id + ".pt"

    try:
      llama_file = self.llama_dir + "chunk" + chunk_id + ".json"
      with open(llama_file, "r") as infile:
        llama_tensors = json.load(infile)
    except:
      index = index // 3000
      chunk = self.chunks[index]
      transcript = chunk["text"]
      start_time = chunk["start_time"]
      end_time = chunk["end_time"]
      m_stime = chunk["mstart_time"]
      r_stime = chunk["rstart_time"]
      chunk_id = chunk["chunk_id"]
      features_path = self.audio_path + "chunk" +  chunk_id + ".pt"

      llama_file = self.llama_dir + "chunk" + chunk_id + ".json"
      with open(llama_file, "r") as infile:
        llama_tensors = json.load(infile)

    trans_length =  end_time - start_time 
    trans_length = trans_length / 1000
    end_time = round_down_to_nearest_02(trans_length)
    end_time_token = f"<|{end_time:.2f}|>"

    m_stime = m_stime - start_time
    m_stime = round_down_to_nearest_02(m_stime/1000)
    mst_token = f"<|{m_stime:.2f}|>"

    if r_stime:
      r_stime = r_stime - start_time
      r_stime = round_down_to_nearest_02(r_stime/1000)
      rst_token = f"<|{r_stime:.2f}|>"

    left, mid, right = transcript.split("|")
    left = left.lstrip()
    right = right.rstrip()

    if left and left.strip():
      mid = mst_token + mst_token + mid
    if r_stime:
      mid = mid + rst_token + rst_token

    transcript = left+ " " + mid+ " " + right
    transcript = transcript.strip()
    transcript = transcript.replace("  ", " ")

    transcript = "<|0.00|>" + transcript + end_time_token + end_time_token
    #print(transcript)
    input_ids = self.tokenizer(transcript, truncation=True, padding='max_length', max_length=128).input_ids
    input_ids = input_ids[:1] + input_ids[2:]

    features = torch.load(features_path)

    '''
    llama_file = self.llama_dir + "chunk" + chunk_id + ".json"
    with open(llama_file, "r") as infile:
      llama_tensors = json.load(infile)
    '''
    context_embed = llama_tensors["context_embeddings"][self.window_index]

    llama_file2 = self.llama_dir + "logits" + chunk_id + ".json"
    with open(llama_file2, "r") as infile:
      llama_tensors = json.load(infile)
    logits = torch.tensor(llama_tensors["logits"], dtype=torch.int8)
    scale_factors = torch.tensor(llama_tensors["scale_factors"], dtype=torch.float32)
    dequantized_logits = (logits.float() * scale_factors)
    chunk_text_embeds = dequantized_logits.squeeze(0)


    item = {
        "labels": input_ids,
        "input_features": features.squeeze(0),
        "context_embeddings": torch.tensor(context_embed, dtype=torch.float32).squeeze(0),
        "chunk_text_logits": chunk_text_embeds,
    }
    return item

