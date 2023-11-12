import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from torch.utils.data import Dataset
import re
import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

# from datasets import load_dataset

# dataset_train = load_dataset("bookcorpus", split='train')

max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
block_size = 256
batch_size = 32
n_head = 7
n_layer = 7
n_embd = 400
dropout = 0.15
model_name = "temp"

with open('inputs.txt', 'r', encoding='utf-8') as f:
    text = f.read()

cleaned_text = [idx for idx in text if not re.findall("[^\u0000-\u05C0\u2100-\u214F]+", idx)]
cleaned_text = "".join(cleaned_text)
cleaned_text = re.sub(r'<(.*)>', 'Thomas', cleaned_text)
cleaned_text = re.sub(r'Surname', 'Thomas', cleaned_text)
cleaned_text = re.sub(r'Forename', 'Benzshawel', cleaned_text)

if torch.cuda.is_available():
    torch.cuda.init()


enc = tiktoken.get_encoding("cl100k_base")
vocab_size = enc.n_vocab

encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

# encoded_data = torch.tensor(data, dtype=torch.long)

#train and validation split
# data = torch.tensor(data, dtype=torch.long)



class CustomTextDataset(Dataset):
    def __init__(self, text):
        self.data = torch.tensor(encode(text), dtype=torch.long)

        n = int(0.9 * len(self.data))

        train_data = self.data[:n]
        val_data = self.data[n:]

        self.train_data = train_data
        self.val_data = val_data

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        # x, y = x.to(device), y.to(device)
        return x, y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.get_batch("train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    train(0, args)


def train(gpu, args):
    torch.manual_seed(0)
    model = BigramLanguageModel()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    #loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = CustomTextDataset(text)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    start = datetime.now()
    # total_step = len(train_loader)

    # making our own criterion
    # @torch.no_grad()
    # def estimate_loss():
    #     out = {}
    #     model.eval()
    #
    #     for split in ['train', 'val']:
    #         losses = torch.zeros(eval_iters)
    #         for k in range(eval_iters):
    #             X, Y = get_batch(split, "text")
    #             logits, loss = model(X, Y)  # <- Problem
    #
    #             # average the loss
    #             if torch.cuda.device_count() > 1:
    #                 temp_loss = torch.zeros(torch.cuda.device_count())
    #                 for i in range(torch.cuda.device_count()):
    #                     temp_loss[i] = loss[i].item()
    #
    #                 losses[k] = temp_loss.mean()
    #             else:
    #                 losses[k] = loss.item()
    #         out[split] = losses.mean()
    #     model.train()
    #     return out

    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (xb, yb) in enumerate(train_loader):

            # if epoch % eval_interval == 0:
            #     losses = estimate_loss()
            #     print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb = xb.cuda(non_blocking=True)
            yb = yb.cuda(non_blocking=True)

            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    total_step,
                    loss.item())
                )

    if gpu == 0:
        print('Training Complete in:', str(datetime.now() - start))


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.droput = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # Single head self attention
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float(
            '-inf'))  # this is a decoder, this means that if you remove this, it becomes an encoder, aka, it can talk to nodes in the future
        wei = F.softmax(wei, dim=-1)
        wei = self.droput(wei)
        v = self.value(x)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.SiLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.fwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.fwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embeding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embeding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx



if __name__ == '__main__':
    main()