import torch
import numpy as np

class MyAddDataSet(torch.utils.data.Dataset):
    '''
    Dataset for transformer models.
    '''
    def __init__(self, func, C=59, diff_vocab=False, eqn_sign=False, device='cpu'):
        self.func = func
        dim = 2
        self.dim = dim
        self.C = C
        self.inputs = []
        self.outputs = []
        self.vocab=C
        if diff_vocab:
            self.vocab*=2
        if eqn_sign:
            self.vocab+=1
            self.dim+=1
        self.vocab_out=0
        for p in range(C**dim):
            x = np.unravel_index(p, (C,)*dim)
            o=self.func(x)
            s=[x[0],x[1]]
            if diff_vocab:
                s[1]+=C
            if eqn_sign:
                s.append(self.vocab-1)
            self.inputs.append(s)
            self.outputs.append(o)
            self.vocab_out=max(self.vocab_out, o+1)
        if self.vocab_out!=C:
            print(f'warning {self.vocab_out=} neq to {C=}')
        self.inputs = torch.tensor(self.inputs, dtype=torch.long, device=device)
        self.outputs = torch.tensor(self.outputs, dtype=torch.long, device=device)
        # print(self.inputs,self.outputs)
    def __len__(self):
        return len(self.outputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

class MyDataset(torch.utils.data.Dataset):
    '''
    Dataset for linear models.
    '''
    def __init__(self, n_vocab=59, device='cpu'):
        self.data = []
        self.n_vocab = n_vocab
        self.device = device
        for i in range(n_vocab):
            for j in range(n_vocab):
                self.data.append([i,j])
    def __getitem__(self, index):
        return torch.tensor(self.data[index],dtype=int),sum(self.data[index])%self.n_vocab
    def __len__(self):
        return len(self.data)