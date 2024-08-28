import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

BATCH_SIZE=64
BLOCK_SIZE=256
LRATE = 3e-4
EMBED_DIM = 384 
DROPOUT = 0.2
NUM_LAYERS = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = torch.device("mps") if torch.backends.mps.is_available() else 'cpu'
print(f"== DEVICE {device} ==")

'''
class LayerNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.ones(dim)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = (self.gamma * xhat) + self.beta

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
'''

# SIMPLE MLP AFTER ATTENTION BLOCK
class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # in the attention is all you need paper the middle layer had a dimension 4 times the input embedding dimension
        self.l1 = nn.Linear(embed_dim, 4*embed_dim, bias=False)
        self.l2 = nn.Linear(4*embed_dim, embed_dim, bias=False)
        self.rectifier = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)

        self.seqnet = nn.Sequential(
                self.l1,
                self.rectifier,
                self.l2,
                self.dropout
        )

    def forward(self, x):
        # self.l2 is the projection layer going back into the residual pathway
        return self.seqnet(x) # self.l2(self.rectifier(self.l1(x)))

# SINGLE HEAD OF ATTENTION
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.head_size = head_size
        
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B,T,C = x.shape

        K = self.key(x) # (B, T, head_size)
        Q = self.query(x) # (B, T, head_size)
        weights = Q @ K.transpose(-2, -1) * (self.head_size ** -0.5) # (B, T, T)

        # weights = torch.zeros((T, T))
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        V = self.value(x)
        out = weights @ V # (B, T, head_size)

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # IGNORE THIS => the Head() network contains token embeddings in the key-query space not the original embedding space
        # projection layer added because its the projection back into the residual pathway
        out = self.projection(out)
        out = self.dropout(out)

        return out

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_size = embed_dim // num_heads
        self.self_attention_layers = MultiHeadAttention(num_heads, head_size)
        self.multilayer_perceptron = MLP(embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x + *block output* IS A SKIP CONNECTION AKA A RESIDUAL CONNECTION
        x = x + self.self_attention_layers(self.layer_norm1(x))
        x = x + self.multilayer_perceptron(self.layer_norm2(x))
        return x

## SIMPLEST NEURAL NETWORK FOR LANGUAGE MODELING
## BIGRAM LANGUAGE MODEL

# torch.manual_seed(1337)
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # each row is an embedding
        # there are n rows where n is the number of tokens in the defined vocabulary (in this case we have 40 possible tokens so our vocab size is 40)
        self.embedding_table = nn.Embedding(vocab_size, embed_dim)
        # in the beginning of the attention block the Token Embedding, the position of the token in the input sequence is encoded into the original token embedding
        # the position embedding table can do that by setting an embedding vector for each of the positions of the input sequence (which is of length BLOCK_SIZE)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, embed_dim)
        # lm_head is short for language modeling head
        # an LM Head is typically used in transformer-based architectures to map vectors from the embed space to the vocabulary size space
        # its typically done using a linear layer
        # this layer generates logits from the embed space to use for predictions for the next token
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.self_attention_head = Head(embed_dim)

        num_heads = 6
        self.multi_attention_head = MultiHeadAttention(num_heads, embed_dim//num_heads)

        self.multilayer_perceptron = MLP(embed_dim)

        self.transformer_blocks = nn.Sequential(*[Block(embed_dim, num_heads) for _ in range(NUM_LAYERS)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        '''
        self.transformer_blocks = nn.Sequential(
            Block(embed_dim, num_heads),
            Block(embed_dim, num_heads),
            Block(embed_dim, num_heads),
            nn.LayerNorm(embed_dim),
        )
        '''

    def forward(self, idx, targets=None):
        # B here is the batch dimensions
        # T here is the input sequence length aka the block size
        B, T = idx.shape

        # logits is in the dimensions (B, T, C)
        # B (batch) is BATCH_SIZE
        # T (time/block size) is BLOCK_SIZE
        # C (channel) is the vocab_size - length of set of all tokens (in this case all characters in our data)
        token_embeddings = self.embedding_table(idx) # (B, T, embed_dim) the C (channels) here is embed_dim
        # torch.arange returns a tensor of length T with integers from 0 to T-1
        # this line creates the embeddings for the positions 0 to T-1
        positional_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)

        # TOKEN EMBEDDINGS - for each batch for each token: tokens are replaced by the token embedding
        # POSITIONAL EMBEDDINGS - for each position: positions are replaced by their positional embedding (according to their position in the token sequence) 
        # TE is (B, T, embed_dim)
        # PE is (T, embed_dim)
        # pytorch will recognize the shape mismatch and add a batch dimension to the positional embeddings
        # since both these embeddings are eventually of the same size: (B, T, embed_dim)
        # we can encode the positional embedding into the original token embedding by vector addition
        x = token_embeddings + positional_embeddings # (B, T, embed_dim)

        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        '''
        # x = self.self_attention_head(x)
        x = self.multi_attention_head(x)

        x = self.multilayer_perceptron(x) # (B, T, C)
        '''

        logits = self.lm_head(x) # (B, T, vocab_size)
        # logits give us the vectors to get our predictions from our input idx for the next character
        # idx is the input which is a batch of n token sequences and embedding_table() will replace each token in each token sequence with the embedding of the next token prediction for all tokens up till that token (inclusive)

        if targets != None:
            # RESIZING (for the cross entropy funciton)
            # pytorch wants the dimensions to be (B, C, T)
            B, T, C = logits.shape
            # (B, T, C) -> (B*T, C)
            logits = logits.view(B*T, C) # dropping batch dimension so now its just a stack of length 40 arrays for all characters in the initial xb mat
            # (B, T) -> (B*T)
            targets = targets.view(B*T)
    
            # LOSS FUNCTION - to measure quality of predictions
            # ENTROPY H(x) = -Σ p(x)log(p(x))
            # CROSS ENTROPY C(x) = -Σ y_c log(p_c) - summed over all classes c: class_label * log(predicted probability of that class)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss
        
    def generate(self, idx, max_gen_tokens):
        # idx is a batch of *batch_size* selected token sequences 
        # each iteration a new predicted token will be appended to the end of the token sequence
        # print(idx.shape)
        # print(idx)
        # print("----")
        for _ in range(max_gen_tokens):
            # do this to ensure that the input idx does not have more than *BLOCKS_SIZE* tokens because the positional embedding table has an upper limit of BLOCK_SIZE
            cropped_idx = idx[:, -BLOCK_SIZE:]

            # gets the predictions for each token subsequence in each sequence of the batch token sequences
            # also gets the loss of all predictions
            logits, loss = self(cropped_idx)

            # now we need to pluck out the last term in the token sequence to use for next token prediction generation
            # so we essentially for each batch replace each length 8 vector of embeddings in logits with the last embedding in that length 8 vector
            logits = logits[:, -1, :] 
            # print(logits.shape)
            # print(logits)
            # we run softmax on the dimension that contains the last in sequence length 40 vector embeddings we just isolated in the previous step to get the probability distribution to sample from for generating the next predicted token
            probs = F.softmax(logits, dim=-1)
            next_token_index = torch.multinomial(probs, num_samples=1) # now we have a single next token prediction for each vector embedding in each batch
            idx = torch.cat((idx, next_token_index), dim=1)
            # print(idx.shape)
            # print(idx)

        return idx

def get_batch(split):
    dataset = train_data if split == "train" else val_data 
    # generates n random idxs in the training dataset where n is the batch size
    # generates from 0 to number of tokens in the dataset minus the block size to ensure no index out of bounds errors
    starts = torch.randint(len(dataset) - BLOCK_SIZE, (BATCH_SIZE,))
    # print(starts)
    # print([dataset[start: start + BLOCK_SIZE] for start in starts]) 
    x = torch.stack([dataset[start: start + BLOCK_SIZE] for start in starts])
    y = torch.stack([dataset[start+1: (start+1) + BLOCK_SIZE] for start in starts])

    return x, y

df = pd.read_csv("./data/updated_rappers.csv")
text = "\n".join(list(df["lyric"]))
vocab_size = len(set(text))

## TOKENIZATION FUNCTIONS
charset = sorted(list(set(text)))
slookup = {s: i for i, s in enumerate(charset)}
ilookup = {i: s for i, s in enumerate(charset)}
stoi = lambda x: [slookup[c] for c in x]
itos = lambda x: "".join([ilookup[num] for num in x])

## PREPROCESSING TEXT DATASET
data = torch.tensor(stoi(text), dtype=torch.long)
print("== DATASET SAMPLE ==")
print(data.shape, data.dtype)
print(data[:100])

# TRAIN TEST SPLIT
train_split = 90
split_idx = int(len(data) * (train_split/100))
train_data = data[:split_idx]
val_data = data[split_idx:]

# MODEL DEFINITION
BLM = BigramLanguageModel(vocab_size, EMBED_DIM)

## TRAINING
# pytorch optimization object
# the learning rate is kind of high and we can get away with that because the the BigramLanguageModel is a very small network
optimizer = torch.optim.AdamW(BLM.parameters(), lr=LRATE)


## changing batch size from 4 to 32
BATCH_SIZE = 32

elosses = []
losses = []
for epoch in range(10000):
    # gets a batch of training data
    xb, yb = get_batch('train')

    # evaluation, gradient calculation, optimization
    logits, loss = BLM(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"EPOCH {epoch} | LOSS {loss.item()}")
        elosses.append(loss.item())
    losses.append(loss.item())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
ax1.plot(losses)
ax1.set_title('All Losses')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross Entropy Loss')

ax2.plot(elosses)
ax2.set_title('Epoch Losses') 
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Cross Entropy Loss')
plt.show()

# POST TRAINING GENERATION
idx = torch.zeros((1, 1), dtype=torch.long)
generated = BLM.generate(idx, 1000)
print("GENERATED_DATA")
print(generated.shape)
print(itos(generated[0].tolist()))
