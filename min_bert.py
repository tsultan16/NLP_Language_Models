"""
    A minimal BERT Model Implementation

    Author: Tanzid Sultan 
"""

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import math
torch.manual_seed(1234)

# use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu' 


class MultiHeadAttention(nn.Module):
    def __init__(self, block_size, embedding_dim, total_head_size, num_heads, dropout_rate):
        super().__init__()

        assert total_head_size % num_heads == 0, "head_size needs to be integer multiple of num_heads"

        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.total_head_size = total_head_size 
        self.head_size = total_head_size // num_heads 
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # define parameters
        self.key = nn.Linear(embedding_dim, self.total_head_size, bias=False)
        self.query = nn.Linear(embedding_dim, self.total_head_size, bias=False)
        self.value = nn.Linear(embedding_dim, self.total_head_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout_rate)

        # we also need to apply a linear projection to make the output residual the same dimension as the input
        self.proj = nn.Linear(total_head_size, embedding_dim) 
        self.output_dropout = nn.Dropout(dropout_rate)


    # define forward pass, input shape: (B,T,C) where B=batch size, T=block_size, C=embedding_dim
    # the attn_mask is a mask that can be used for masking out the attention weights for padding tokens 
    def forward(self, x, attn_mask):
        B, T, C = x.shape
        #print(f"B = {B}, T={T}, C={C}")
        k = self.key(x) # (B,T,H) where H is the total_head_size
        q = self.query(x) # (B,T,H)
        v = self.value(x) # (B,T,H)

        # reshape (B,T,H) --> (B,T,n,h), where n=num_heads and h=head_size and H=n*h
        k = k.view(B,T,self.num_heads,self.head_size) 
        q = q.view(B,T,self.num_heads,self.head_size) 
        v = v.view(B,T,self.num_heads,self.head_size) 

        # now we transpose so that the num_heads is the second dimension followed by T,h
        # this allows us to batch matrix mutliply for all heads simulataneously to compute their attention weights
        # (B,T,n,h) --> (B,n,T,h) 
        k = k.transpose(1,2) 
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # compute attention scores manually (slower)
        W = q @ k.transpose(-2,-1)  / math.sqrt(self.head_size) # (B,n,T,T)
        attn_mask = attn_mask.view(B,1,1,T)        
        #print(f"W shape= {W.shape}, attn_mask shape = {attn_mask.shape}")
        W = W.masked_fill(attn_mask == 0, float('-inf')) 
        W = F.softmax(W, dim=-1)
        # apply dropout to attention weights
        W = self.attn_dropout(W)
        out = W @ v # (B,n,T,h)
        

        # use pytorch built-in function for faster computation of attention scores (set the 'is_causal' parameter for applying causal masking)
        #out = F.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask.bool(),dropout_p=self.dropout_rate if self.training else 0,is_causal=False)

        # we can transpose the output from (B,n,T,h) --> (B,T,n,h)
        # since the last two dimensions of the transposed tensor are non-contiguous, we apply 
        # contiguous() which return a contiguous tensor
        out = out.transpose(1,2).contiguous()

        # finally we collapse the last two dimensions to get the concatenated output, (B,T,n,h) --> (B,T,n*h) 
        out = out.view(B,T,self.total_head_size)

        # now we project the concatenated output so that it has the same dimensions as the multihead attention layer input
        # (we need to add it with the input because of the residual connection, so need to be same size) 
        out = self.proj(out) # (B,T,C) 

        # apply dropout
        out = self.output_dropout(out)

        return out
    

# a simple mlp 
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout_rate):
        super().__init__()
        # we add extra computations by growing out the feed-forward hidden size by a factor of 4
        # we also add an extra linear layer at the end to project the residual back to same dimensions as input
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4*embedding_dim),  
            nn.ReLU(),
            nn.Linear(4*embedding_dim, embedding_dim), 
            nn.Dropout(dropout_rate)
        )
    
    # in the forward pass, concatenate the outputs from all the attention heads
    def forward(self, x):
        return self.net(x)
    

# transformer encoder block with residual connection and layer norm
# Note: the original transformer uses post layer norms, here we use pre layer norms, i.e. layer norm is applied at the input
# instead of the output, this typically leads to better results in terms of training convergence speed and gradient scaling 
class TransformerBlock(nn.Module):
    def __init__(self, block_size, embedding_dim, head_size, num_heads, dropout_rate):
        super().__init__()
        self.sa = MultiHeadAttention(block_size, embedding_dim, head_size, num_heads, dropout_rate) # multi-head attention layer 
        self.ff = FeedForward(embedding_dim, dropout_rate)   # feed-forward layer
        self.ln1 = nn.LayerNorm(embedding_dim) # layer norm at input of multi-head attention
        self.ln2 = nn.LayerNorm(embedding_dim) # layer norm at input of feed-forward

    # in the forward pass, concatenate the outputs from all the attention heads
    def forward(self, x, attn_mask):
        # residual connection between input and multi-head attention output (also note that we're doing a pre-layer norm, i.e. layer norm at the input of the multi-head attention)
        x = x + self.sa(self.ln1(x), attn_mask)
        # residual connection between multi-head attention output and feed-forward output (also note that we're doing a pre-layer norm, i.e. layer norm at the input of the feed-forward)
        x = x + self.ff(self.ln2(x)) 
        return x
    

# BERT model with multiple transformer blocks 
class BERTModel(nn.Module):
    def __init__(self, vocab_size, block_size, embedding_dim, head_size, num_heads, num_blocks, pad_token_id, dropout_rate=0.2):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size        # block_size is just the input sequence length
        self.embedding_dim = embedding_dim
        self.head_size = head_size
        self.hum_heads = num_heads
        self.num_blocks = num_blocks

        '''
        Define model parameters
        '''
        # token embedding layer 
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id) # shape: (vocab_size,C)
        # position embedding layer
        self.pos_embedding = nn.Embedding(block_size, embedding_dim) # shape: (T,C)
        # segment embedding layer (disabled for now)
        #self.segment_embedding = nn.Embedding(2, embedding_dim)

        # stack of transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(block_size, embedding_dim, head_size, num_heads, dropout_rate) for _ in range(num_blocks)])

        # pooling transformation of CLS token (for downstream tasks requiring full sentence hidden representation)
        #self.pooling_linear = nn.Linear(embedding_dim, embedding_dim) # shape: (C,C)
        #self.pooling_activation_fn = nn.Tanh()

        # output layer
        self.ln = nn.LayerNorm(embedding_dim)
        self.output_linear = nn.Linear(embedding_dim, vocab_size)
        self.dropout = torch.nn.Dropout(dropout_rate)

        # store position indices inside a buffer for fast access when computing position embeddings
        position_idx = torch.arange(block_size, device=device).unsqueeze(0)
        self.register_buffer('position_idx', position_idx)


        # forward pass takes in a batch of input token sequences idx of shape (B,T) and corresponding targets of shape (B,T)
    def forward(self, idx, attn_mask, segment_idx=None):
        B, T = idx.shape
        # get token embeddings
        token_embeds = self.token_embedding(idx) # (B,T,C)
        # add positional encoding
        pos_embeds = self.pos_embedding(self.position_idx[:,:T]) # (T,C) 
        
        # add sentence segment embedding (disabled for now)
        # segment_embeds = self.segment_embedding(segment_idx) # segment_idx is an integer tensor of shape (B,T) and has 0's at positions corresponding to 
        
        # the first sentence and 1's at positions corresponding to the second sentence 
        x = token_embeds + pos_embeds # (B,T,C)
        # pass through transformer blocks to get encoding
        for block in self.blocks:
            x = block(x, attn_mask) # (B,T,C)
    
        # get CLS token encoding and apply pooling transform
        #cls_encoding = x[:,0] # (B,C)
        #pooled_cls_encoding = self.pooling_activation_fn(self.pooling_linear(cls_encoding)) # (B,C)

        # apply final layers norm
        x = self.ln(x)

        # compute output logits
        logits = self.output_linear(self.dropout(x))

        return logits 
