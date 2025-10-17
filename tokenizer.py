import torch 
import tiktoken
import random

'''
Steps: 
1. Load the dataset 
2. Batching and making blocks of the dataset
3. Make int
'''

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
        self.batch_size = 10
        self.data_size = len(self.data)
        self.enc = tiktoken.get_encoding("o200k_base")

        self.data_tokenized = torch.tensor(self.enc.encode(self.data)).to(self.device)

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

        start_points = torch.randint(0,self.vocab_size-self.block_size,(self.batch_size,))
        print(start_points)
        X = torch.stack([self.data_tokenized[i:i+self.block_size] for i in start_points])
        y = torch.stack([self.data_tokenized[i+1:i+self.block_size+1] for i in start_points])
        return X,y
        
        

    def tokenize(self,input_text): 
        ## Tokenization ==> Using Tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        assert enc.decode(enc.encode("hello world")) == "hello world"
        print(enc.encode("Bharat Jain"))


def main(): 
    gpt = Nano_GPT()
    encoded = gpt.simple_encoding("Bharat Jain")
    print(encoded, "\n", gpt.simple_decoding(encoded))
    print(gpt.get_batch())
if __name__ == "__main__": 
    main()




