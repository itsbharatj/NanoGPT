import torch 
import tiktoken
import random
import torch.nn as nn 
from torch.nn import functional as F 
torch.manual_seed(1337)
import torch
'''
Steps: 
1. Load the dataset 
2. Batching and making blocks of the dataset
3. Make int
'''
if torch.cuda.is_available(): 
    device = "cuda" 
elif torch.backends.mps.is_available(): 
    device = "mps" 
else: 
    device = "cpu"

block_size = 8 
batch_size = 32
n_embed = 32

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size): 
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
        self.position_embedding_table = nn.Embedding(block_size,n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size)
    
    def forward(self, idx, target=None): 
        B,T = idx.shape

        tok_emd = self.token_embedding_table(idx) ## Put the embeddings from this batch (B,T,C->vocab_size) ==> (4,8,64)
        pos_emd = self.position_embedding_table(torch.arange(T,device=device)) ## Creates an (T,C)
        x = tok_emd+pos_emd
        logits = self.lm_head(x) # (B,T,C)
        if target is None: 
            loss = None
        else: 
            B,T,C = logits.shape
            logits_R = logits.view(B*T,C)
            target = target.view(B*T)

            loss = F.cross_entropy(logits_R,target)

        return logits,loss

    def generate(self,idx,max_num_tokens): 
        for _ in range(max_num_tokens): 
            ## Get the predictions: 
            logits,loss = self(idx) ## runs the forward functions by default

            ## focus only on the last token: 
            logits = logits[:,-1,:]

            ## apply softmax to get probs: 
            probs = F.softmax(logits,dim=1) ## (B,C)

            ## Sample from the distribution: 
            idx_next = torch.multinomial(probs,num_samples=1) ## Does it also give the original one?

            ## Append the sampled idx to the running sequence: 
            idx = torch.cat((idx,idx_next),dim=1)
        
        return idx


class Nano_GPT: 
    def __init__(self):
        self._load_dataset()

        ## Constants: 
        if torch.cuda.is_available():
            self.device = "cuda" 
        elif torch.backends.mps.is_available(): 
            self.device = "mps" 
        else: 
            self.device = "cpu"

        self.block_size = 8 
        self.batch_size = 32
        self.data_size = len(self.data)
        # self.enc = tiktoken.get_encoding("gpt-2")

        self.data_tokenized = torch.tensor(self.simple_encoding(self.data)).to(self.device)
        print(len(self.data_tokenized))
        self.model = BigramLanguageModel(self.vocab_size).to(self.device)  # Use tiktoken vocab size, not character vocab!
        X,y = self.get_batch()
        print(X,y)

        logits,loss = self.model(X,y)
        print(f'Loss: {loss}')

        self.optim = torch.optim.AdamW(self.model.parameters())


        #90-10 train val split
        split = int(self.data_size*0.90)
        train_data = self.data[:split]
        val_data = self.data[split:]
        
    
    def _load_dataset(self,path=None): 
        data = "input.txt" if path is None else path    
        with open(data) as f: 
            self.data = f.read()
        self.data_size = len(self.data)
        self.s_vocab = sorted(list(set(self.data))) 
        self.vocab_size = len(self.s_vocab)
        self.vocab_map = {c:ind for ind,c in enumerate(self.s_vocab)}
        print(f"Size of the vocab {self.vocab_size}\n Elements: {self.s_vocab}")
        
    
    def simple_encoding(self,input_text): 
        ## Using a simple encoding based on the charater index --> character mapping and vice_versa 
        ## Every char is a token, so the context of just the char may not be enough 
        encoded = [self.vocab_map[c] for c in input_text]
        return encoded

    def simple_decoding(self,encoded_text): 
        decoded = "".join([self.s_vocab[i] for i in encoded_text])
        return decoded

    def get_batch(self): 
        ## Get one batch of the shakespeare dataset, across two dimensions: Batch dimension and time dimension
        ## One batch is on a random snippet of the data for each batch here

        start_points = torch.randint(0,len(self.data_tokenized)-self.block_size,(batch_size,))
        X = torch.stack([self.data_tokenized[i:i+self.block_size] for i in start_points]).to(self.device)
        y = torch.stack([self.data_tokenized[i+1:i+self.block_size+1] for i in start_points]).to(self.device)
        return X,y
        
        

    def tokenize(self,input_text): 
        ## Tokenization ==> Using Tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        assert enc.decode(enc.encode("hello world")) == "hello world"
        print(enc.encode("Bharat Jain"))
    
    def trainer(self,epochs=10000): 
        for steps in range(epochs): 
            xb,yb = self.get_batch()
            logits, loss = self.model(xb, yb)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()
        print(loss.item())


def main(): 
    gpt = Nano_GPT()
    encoded = gpt.simple_encoding("Bharat Jain")
    print(encoded, "\n", gpt.simple_decoding(encoded))
    print(gpt.simple_decoding(gpt.model.generate(torch.zeros((1,1),dtype=torch.long).to(gpt.device),100)[0].tolist()))
    gpt.trainer()
    print(gpt.simple_decoding(gpt.model.generate(torch.zeros((1,1),dtype=torch.long).to(gpt.device),1000)[0].tolist()))


if __name__ == "__main__": 
    main()