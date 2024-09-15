import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizer import AudioTokenizer
from constants import SEP_TOKEN_ID


class LJSpeechDataset(Dataset):
    def __init__(self, split="train"):
        self.dataset = load_dataset("keithito/lj_speech", trust_remote_code=True)[split]

        self.text_tokenizer = AutoTokenizer.from_pretrained("mayura-ai/sarika")
        self.audio_tokenizer = AudioTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datapoint = self.dataset[idx]

        text_tokens = torch.tensor(
            self.text_tokenizer.encode(datapoint["normalized_text"])
        )
        audio_tokens = self.audio_tokenizer.encode(
            torch.tensor(datapoint["audio"]["array"], dtype=torch.float32).unsqueeze(0)
        )

        bos_token = torch.tensor([self.text_tokenizer.bos_token_id])
        eos_token = torch.tensor([self.text_tokenizer.eos_token_id])
        sep_token = torch.tensor([self.text_tokenizer.sep_token_id])

        concatenated_tokens = torch.cat(
            (bos_token, text_tokens, sep_token, audio_tokens.squeeze(0), eos_token)
        )

        return concatenated_tokens


def collate_fn(batch):
    print(batch, end="========")
    X = []
    Y = []
    for item in batch:
        split_index = torch.where(item == SEP_TOKEN_ID)[0][0]
        X.append(item[: split_index + 1])
        Y.append(item[split_index + 1 :])
    fin_X = []
    fin_Y = []
    for x, y in zip(X, Y):
        for i in range(len(y) - 1):
            fin_X.append(torch.cat((x, y[:i])))
            fin_Y.append(y[i:])
    return torch.nn.utils.rnn.pad_sequence(fin_X), torch.nn.utils.rnn.pad_sequence(
        fin_Y
    )
