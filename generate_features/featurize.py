from pydub import AudioSegment
import os
import torch

from transformers import AutoFeatureExtractor,
import json, librosa

feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")



with open("corrected_all.jsonl", "r") as injs:
  for line in injs:
    chunk_dict = json.loads(line)
    filename = chunk_dict["filename"]
    chunks = chunk_dict["chunks"]
    try:
      audio_path = "../dataset/english/" + filename + "/audio.ogg"
      audio, original_sr = librosa.load(audio_path, sr=None)  # sr=None preserves the original rate
    except:
      continue
    for chunk_js in chunks:
        if chunk_js:
          chunk_id = str(chunk_js["global_chunk_number"])
          lwindow, mwindow, rwindow = chunk_js["left"], chunk_js["mid"], chunk_js["right"]
          cstart = lwindow["start_time"] or mwindow["start_time"]
          cend = rwindow["end_time"] or mwindow["end_time"]
          chunk_no = chunk_js["global_chunk_number"]

          start_sample = int((cstart/ 1000) * original_sr)
          end_sample = int((cend / 1000) * original_sr)

          audio_segment = audio[start_sample:end_sample]

          target_sr = 16000
          segment_resampled = librosa.resample(audio_segment, orig_sr=original_sr, target_sr=target_sr)
          input_features = feature_extractor(segment_resampled, sampling_rate=16000, return_tensors="pt").input_features
          #print(input_features["input_features"])
          ofilename = "feats/chunk" + str(chunk_no) + ".pt"
          print(chunk_no)
          torch.save(input_features, ofilename)

