"""
    WordPiece Tokenizer Implementation

    Author: Tanzid Sultan    
"""

from collections import defaultdict
import string, re
import unicodedata
from multiprocess import Pool
from tqdm import tqdm
import random 
random.seed(1234)


class WordPieceTokenizer():
    def __init__(self, cleaning=False, max_subword_len=20):
        self.cleaning = cleaning
        self.vocab = []
        self.word2int = {}
        self.int2word = {}
        # special tokens
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        
        self.max_subword_len = max_subword_len
        self.invalid_chars = ('*', '~', '_', '^', '`', '+', '\\', '[', ']', '<', '>')
        self.max_vocab_size = None

        

    def mask_token_id(self):
        return self.word2int[self.mask_token]

    def pad_token_id(self):
        return self.word2int[self.pad_token]

    def cls_token_id(self):
        return self.word2int[self.cls_token]

    def unk_token_id(self):
        return self.word2int[self.unk_token]

    def sep_token_id(self):
        return self.word2int[self.sep_token]

    def vocab_size(self):
        return len(self.vocab)
    
    def clean_sentence(self, s):
        if self.cleaning:
            # removes all control characters and invalid characters, replaces multiple adjacent whitespace with single whitespace
            s = "".join(ch for ch in s if unicodedata.category(ch)[0] != 'C' and ch not in self.invalid_chars) 
            s = " ".join(s.split())
            
            # remove all non-letter characters
            #s = "".join(ch for ch in s if unicodedata.category(ch)[0]=='L' or unicodedata.category(ch)=='Zs') 
            #s = " ".join(s.split())        
        return s

    # generates wordpiece vocabulary of subwords from a given corpus
    # the input corpus is a list of sentences
    def generate_vocab(self, corpus, max_vocab_size):
        self.max_vocab_size = max_vocab_size
        # pretokenize the corpus into words and get unigram counts, we will only use the first sentence as an example
        word_freqs = defaultdict(int)
        for s in corpus:
            s = self.clean_sentence(s)
            words = s.split()
            for word in words:
                word_freqs[word] += 1
        
        # initialize WordPiece vocabulary
        for word in word_freqs.keys():
            if word[0] not in self.vocab:
                self.vocab.append(word[0])
            for letter in word[1:]:
                prefixed = '##' + letter
                if prefixed not in self.vocab:
                    self.vocab.append(prefixed)

        # now add special tokens
        self.vocab = self.vocab + [self.pad_token, self.cls_token, self.unk_token, self.mask_token, self.sep_token]

        # generate splits
        splits = {word: [c if i==0 else f"##{c}" for i,c in enumerate(word)] for word in word_freqs.keys()}
        
        # function for computing pair scores
        def compute_pair_scores(splits):
            letter_freqs = defaultdict(int)
            pair_freqs = defaultdict(int)

            for word,freq in word_freqs.items():            
                split = splits[word]
                # if word only contains one split
                if len(split) == 1:
                    letter_freqs[split[0]] += freq
                    continue
                
                # count up every individual split and adjacent pair of splits 
                for i in range(len(split)-1):
                    pair = (split[i], split[i+1])
                    letter_freqs[split[i]] += freq

                    # apply penalty to pairs that exceed max_subword_len 
                    # or splits containing numeric characters so they won't get merged
                    if (len(split[i]+ split[i+1]) > self.max_subword_len) or re.search(r'\d', split[i]) or re.search(r'\d', split[i+1]):
                        pair_freqs[pair] =0
                    else:
                        pair_freqs[pair] += freq

                letter_freqs[split[-1]] += freq

            scores = {pair: freq/(letter_freqs[pair[0]]*letter_freqs[pair[1]]) for pair,freq in pair_freqs.items()}
            return scores
        
        # function for merging a pair of splits    
        def merge_pair(c1, c2, splits):
            for word in word_freqs:
                split = splits[word]
                if len(split) > 1:
                    i = 0
                    while i < len(split)-1:
                        if split[i] == c1 and split[i+1] == c2:
                            merged = c1 + c2.lstrip('#')
                            split = split[:i] + [merged] + split[i+2:]
                        else:
                            i += 1
                        splits[word] = split

            return splits    
        
        # generate the subword vocabulary
        pbar = tqdm(total=max_vocab_size, desc="Building vocab. Current vocab_size --> ")
        pbar.update(len(self.vocab))

        while len(self.vocab) < max_vocab_size:
            # compute all pair scores
            pair_scores = compute_pair_scores(splits)
            # get pairs with largest score 
            max_score = max(pair_scores.values())
            max_score_pairs = [pair for pair, score in pair_scores.items() if score== max_score]
            # randomly break ties
            max_score_pair = random.choice(max_score_pairs)
            # add new subword to vacabulary
            subword = max_score_pair[0] + max_score_pair[1].lstrip('#')
            self.vocab.append(subword)
            # update splits 
            splits = merge_pair(*max_score_pair, splits)
            pbar.update(1)
            
        self.vocab = sorted(set(self.vocab))
        self.word2int = {word:i for i,word in enumerate(self.vocab)}
        self.int2word = {i:word for i,word in enumerate(self.vocab)}


    def encode_sentence(self, s):
        # first clean the sentence
        s = self.clean_sentence(s)
        # tokenize the sentence into subword sequence
        subword_tokens = self.tokenize_sentence(s)
        # convert to token indices
        indices = [self.word2int[t] for t in subword_tokens]
        return indices

    # encode sentence into subword token indices
    def encode(self, sentences):
        encoded_sentences = []
        with Pool() as pool:
            # Note: since we're using map instead of imap, order of encoded sequences will differ from order of original sequences
            for result in tqdm(pool.map(self.encode_sentence, sentences), total=len(sentences), desc="Encoding sequences."):
                encoded_sentences.append(result)
        return encoded_sentences


    def decode_indices(self, indices):
        # first cnvert indices to subword tokens
        subwords = [self.int2word[ix] for ix in indices]
        # merge subwords
        i = 0
        while i < len(subwords)-1:
            a = subwords[i]
            b = subwords[i+1]
            if len(b) == 1:
                i += 1  
                continue
            if b[:2]=="##":
                subwords = subwords[:i] + [a+b.lstrip('#')] + subwords[i+2:]
            else:       
                i += 1    
        s = " ".join(subwords)
        return s


    # decode subword token index sequences back to sentences
    def decode(self, idx):
        sentences = []
        with Pool() as pool:
            for result in tqdm(pool.imap(self.decode_indices, idx), total=len(idx), desc="Decoding sequences."):
                sentences.append(result)    
        return sentences


    def tokenize_sentence(self, sent):
        tokens = []
        # split the sentence into words 
        # make sure to convert all characters to lower case because our vocabulary does not contain
        # upper case letters
        words = sent.lower().split()
        # tokenize each word
        for word in words:
            tokens = tokens + self.tokenize_word(word)
        return tokens

    def tokenize_word(self, word):
        tokens = []
        while len(word) > 0:
            i = len(word)    
            # find longest mactching subword subword
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                # no match found
                return [self.unk_token]
            # found longest subword
            tokens.append(word[:i])
            word = word[i:]
            # add prefix
            if len(word) > 0:
                word = f"##{word}"
        return tokens          
