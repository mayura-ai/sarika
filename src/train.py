import torch
import torch.nn as nn
from model import Sarika, SarikaConfig
from data import LJSpeechDataset, collate_fn
from tqdm import tqdm
from torch.utils.data import DataLoader
from constants import SEP_TOKEN_ID

dataset = LJSpeechDataset()
config = SarikaConfig(vocab_size=91235)
model = Sarika(config=config)

num_epochs = 10
device = "mps"
optim = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
model = model.to(device)

eos_token_id = dataset.text_tokenizer.eos_token_id


batch_size = 1
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)

model.train()
pbar = tqdm(range(num_epochs))

# TODO: Fix training loop
for epoch in pbar:
    epoch_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        sep_token_index = batch.eq(SEP_TOKEN_ID).nonzero()[:, 1]
        X = batch[:, : sep_token_index[0] + 1]
        Y = batch[:, sep_token_index[0] + 1 :]

        Y_hat = model(X)

        Y_hat = Y_hat[:, -Y.size(1) :, :]
        Y_hat = Y_hat.contiguous().view(-1, Y_hat.size(-1))
        Y = Y.contiguous().view(-1)

        loss = criterion(Y_hat, Y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    pbar.set_description(f"Epoch: {epoch+1}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "./weights.pth")
