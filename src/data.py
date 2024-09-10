# Example data

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from audio.tokenizer import SNACTokenizer


text_tokenizer = AutoTokenizer.from_pretrained("gpt2")
audio_tokenizer = SNACTokenizer()

dataset = load_dataset("keithito/lj_speech", trust_remote_code=True)
datapoint = dataset["train"][0]

text_tokens = text_tokenizer.encode(datapoint["normalized_text"], return_tensors="pt")
audio_tokens = audio_tokenizer.encode(
    torch.tensor(datapoint["audio"]["array"], dtype=torch.float32).unsqueeze(0)
)

BOS = torch.tensor([420420420], dtype=torch.int32).unsqueeze(0)
EOS = torch.tensor([636363636], dtype=torch.int32).unsqueeze(0)
SEP = torch.tensor([696969696], dtype=torch.int32).unsqueeze(0)

print(text_tokens.shape, audio_tokens.shape, BOS.shape, EOS.shape, SEP.shape)

concatenated_tokens = torch.cat((BOS, text_tokens, SEP, audio_tokens, EOS), dim=1)

print(concatenated_tokens.shape)

print(concatenated_tokens)
