# Whispering Context: Distilling Syntax and Semantics for Long Speech Transcripts

This is the repo for IEEE ASRU 2025 paper [Whispering Context: Distilling Syntax and Semantics for Long Speech Transcripts](). You can reach the preprint at [Arxiv]().

Our work offers a novel approach for blending transcript syntax&semantics into training chunks. We propose a novel approach that enhances ASR by distilling contextual knowledge from LLaMA models into Whisper. We measure syntactic&semantic success by measuring NER success, punctuation and capitalization success.

A key innovation of our approach is the use of an extended context window during representation-level distillation to provide Whisper with richer textual clues. Instead of
distilling knowledge solely from LLaMA’s representation of the chunk text (e.g., a 30-second segment), we incorporate a broader context from the entire transcript. Specifically, for
a fixed context size of tokens, we include half the tokens preceding the chunk and half following it within the transcript, effectively expanding the semantic scope available during
distillation. We experimented with various context sizes—64, 128, 256, 512, and 1024 tokens—and observed that increasing the context size significantly improved NER performance
by enabling Whisper to leverage more semantic information from the surrounding text. This extended context trick proved particularly effective for handling long-tail entities, where
global transcript-level understanding is critical for accurate recognition and formatting. Below figure exhibits an overview of our approach.


<p align="center">
<img src="images/interfinal.png" width=800></img>
</p>

## Running the experiment

To run the experiments

- Generate and persist speech features by `generate_features/featurize.py`
- Generate LlaMa tensors by `generate_llama_tensors/get_vectors_quantized.py`
- Run the experiment by `distill/run.sh`
