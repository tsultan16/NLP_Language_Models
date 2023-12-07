"""
    Pre-Training BERT on Masked Language Modeling Task

    Author: Tanzid Sultan

"""

from torch.utils.data import Dataset, DataLoader
import psutil
import sys
from itertools import islice
import pickle
from min_bert import BERTModel
from wordpiece_tokenizer import WordPieceTokenizer
import torch
from torch import nn
from torch.nn import functional as F
import random 
from tqdm import tqdm
random.seed(1234)

# pre-trainng dataset
class BERTDataset(Dataset):
    def __init__(self, corpus, tokenizer, block_size, mlm_prob=0.15):
        self.corpus = corpus          # encoded sentences
        self.tokenizer = tokenizer    # wordpiece tokenizer
        self.block_size = block_size  # truncation/max length of sentences
        self.corpus_len = len(corpus) # size of corpus
        self.mlm_prob = mlm_prob
        self.vocab_size = tokenizer.vocab_size()

    def __len__(self):
        return self.corpus_len
    

    def __getitem__(self, idx):
        # get the sentence 
        s = self.get_corpus_sentence(idx)
        # truncate to block_size-1
        s = s[:self.block_size-1] 
        s_len = len(s)
        
        # replace tokens randomly
        s, label = self.replace_tokens(s)
        # append the CLS token at the beginning of sentence
        s = [self.tokenizer.cls_token_id()] + s
        # apply padding
        pad_len = max(0,self.block_size-s_len-1)
        s = s + [self.tokenizer.pad_token_id()]*pad_len
        label = [-100] + label + [-100]*pad_len
        # create attention mask which has 0's at positions of pad tokens and 1's everywhere else 
        attention_mask = [1]*(self.block_size-pad_len) + [0]*pad_len       

        # convert to torch tensors
        s = torch.tensor(s)
        label = torch.tensor(label)
        attention_mask = torch.tensor(attention_mask)

        # Note: Unlike the original BERT, we are not returning a pair of sentences, so we
        # don't need to return segment labels or next_sentence label 
        return {"masked_input" : s, "label" : label, "attention_mask" : attention_mask}


    # randomly replace tokens with mlm_prob probability
    def replace_tokens(self, s):
        # the labels for a masked token is the original token index and -100 for non-masked tokens
        label = [-100] * len(s)
        for i,t in enumerate(s):
            p = random.random()    
            if p < self.mlm_prob:
                p = p/self.mlm_prob
                # replace with masked token with 80% probability
                if p < 0.8:
                    s[i] = self.tokenizer.mask_token_id()        

                elif p < 0.9:
                    # replace with random token with 10% probability 
                    s[i] = random.randrange(self.vocab_size)
                
                # Note: for all three cases, i.e. token getting replaced by mask token, token getting replaced
                # by another random token or token not getting replaced, we want to predict the actual word as our target label 
                label[i] = t

        return s, label


    def get_corpus_sentence(self, idx):
        return self.corpus[idx]        
    

def save_model_checkpoint(model, optimizer, epoch=None, loss=None):
    # Save the model and optimizer state_dict
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    # Save the checkpoint to a file
    torch.save(checkpoint, 'BERT_checkpoint.pth')
    print(f"Saved model checkpoint!")

# load pre-trained tokenizer from file
with open('WordPiece_tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file) 

# load pre-encoded sequences (~2M sequences)
with open('dataset_encoded_1', 'rb') as file:
    dataset_encoded = pickle.load(file)

print(f"Num encoded sequences = {len(dataset_encoded)}")

# create dataset
block_size = 96
batch_size = 32
train_dataset = BERTDataset(dataset_encoded, tokenizer, block_size=block_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)  # set pin_memory for faster pre-fetching 
print(f"Total number of batches: {len(train_dataloader)}")


# model hyperparameters
embedding_dim = 384
head_size = embedding_dim
num_heads = 4 #12
num_blocks = 1 #8
dropout_rate = 0.2
max_iters = 1
learning_rate = 1e-4
smoothed_loss = 0.0
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
trained_epochs = 0

load_pretrained = False

# instantiate model
if(load_pretrained):
    
    # Load the checkpoint from the file
    checkpoint = torch.load('BERT_checkpoint.pth')
    # Initialize the model and optimizer
    model = BERTModel(vocab_size=tokenizer.vocab_size(), block_size=block_size, embedding_dim=embedding_dim, head_size=head_size, num_heads=num_heads, num_blocks=num_blocks, pad_token_id=tokenizer.pad_token_id(), dropout_rate=dropout_rate)
    # move model to device
    m = model.to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    # Load the model and optimizer state_dict from the checkpoint
    m.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trained_epochs = checkpoint['epoch']
    smoothed_loss = checkpoint['loss']
    m.train()

else:

    model = BERTModel(vocab_size=tokenizer.vocab_size(), block_size=block_size, embedding_dim=embedding_dim, head_size=head_size, num_heads=num_heads, num_blocks=num_blocks, pad_token_id=tokenizer.pad_token_id(), dropout_rate=dropout_rate)
    # move model to device
    m = model.to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)



num_params = sum(p.numel() for p in m.parameters())
print(f"Total number of parameters in transformer network: {num_params/1e6} M")
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")


# training loop
for epoch in range(max_iters):
    pbar = tqdm(train_dataloader, desc="Epochs")
    for batch in pbar:
        # sample a batch of trainin data
        xb, yb, attn_mask = batch['masked_input'], batch['label'], batch['attention_mask'] 
        # move batches to gpu
        xb = xb.to(device)
        yb = yb.to(device)
        attn_mask = attn_mask.to(device)

        # forward pass
        logits = m(xb, attn_mask)
        # compute loss
        B,T,vocab_size = logits.shape
        # reshape the logits and targets such that batch of input sequences are flattened into a single big input sequence
        # i.e. (B,T) --> (B*T)
        logits = logits.view(B*T,vocab_size) # reshaped to (B*T,vocab_size)
        yb = yb.view(B*T) # reshaped to (B*T)
        # compute cross entropy loss (i.e. average negative log likelihood)
        loss = F.cross_entropy(logits, yb, ignore_index=-100)

        # exponential moving average loss
        smoothed_loss = 0.9 * smoothed_loss + 0.1 * loss

        # reset parameter gradients
        optimizer.zero_grad(set_to_none=True) 
        # backward pass
        loss.backward()
        # optimizer step
        optimizer.step()

        pbar.set_description(f"Epoch {epoch + 1}, Batch Loss: {loss:.3f}, Moving avg. Loss: {smoothed_loss:.3f}")   
    
    # save checkpoint 
    trained_epochs += 1
    save_model_checkpoint(m, optimizer, loss=smoothed_loss, epoch=trained_epochs)    


print(f"Training done!")






