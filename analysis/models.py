from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

d_hidden=256
n_vocab=59

class MyModelD(nn.Module):
    def __init__(self):
        super(MyModelD, self).__init__()
        self.embed = nn.Embedding(n_vocab, d_hidden//2)
        self.unembed = nn.Embedding(n_vocab, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.embed.weight.data /= (d_hidden//2)**0.5
        self.unembed.weight.data /= d_hidden**0.5
        self.pause=False
    def backdoor_h(self, x):
        x = self.embed(x)
        x0 = x
        x = torch.cat([x[:,0],x[:,1]], dim=1)
        x = self.l1(x)
        x = F.relu(x)
        return x0,x
    def forward_h(self, x):
        _, x = self.backdoor_h(x)
        x = x @ self.unembed.weight.t()
        return _, x
    def backdoor(self,x):
        return self.backdoor_h(x)[1]
    def forward(self,x):
        return self.forward_h(x)[1]

class MyModelA(nn.Module):
    def __init__(self):
        super(MyModelA, self).__init__()
        self.embed = nn.Embedding(n_vocab, d_hidden)
        self.unembed = nn.Embedding(n_vocab, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.embed.weight.data /= (d_hidden//2)**0.5
        self.unembed.weight.data /= d_hidden**0.5
    def backdoor_h(self, x):
        x = self.embed(x)
        x0 = x
        assert len(x.shape)==3 and x.shape[1]==2
        x = self.l1(x[:,0]+x[:,1])
        x = F.relu(x)
        #x = self.l2(x)
        #x = F.relu(x)
        return x0,x
    def forward_h(self, x):
        _, x = self.backdoor_h(x)
        x = x @ self.unembed.weight.t()
        return _, x
    def backdoor(self,x):
        return self.backdoor_h(x)[1]
    def forward(self,x):
        return self.forward_h(x)[1]

class MyModelB(nn.Module):
    def __init__(self):
        super(MyModelB, self).__init__()
        self.embed = nn.Embedding(n_vocab, d_hidden)
        self.unembed = nn.Embedding(n_vocab, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.l2 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.embed.weight.data /= (d_hidden//2)**0.5
        self.unembed.weight.data /= d_hidden**0.5
    def backdoor_h(self, x):
        x = self.embed(x)
        x0 = x
        assert len(x.shape)==3 and x.shape[1]==2
        x = self.l1(x[:,0]+x[:,1])
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        return x0, x
    def forward_h(self, x):
        _, x = self.backdoor_h(x)
        x = x @ self.unembed.weight.t()
        return _, x
    def backdoor(self,x):
        return self.backdoor_h(x)[1]
    def forward(self,x):
        return self.forward_h(x)[1]

class MyModelC(nn.Module):
    def __init__(self):
        super(MyModelC, self).__init__()
        self.embed = nn.Embedding(n_vocab, d_hidden)
        self.unembed = nn.Embedding(n_vocab, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.l2 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.embed.weight.data /= (d_hidden//2)**0.5
        self.unembed.weight.data /= d_hidden**0.5
    def backdoor_h(self, x):
        x = self.embed(x)
        x0 = x
        assert len(x.shape)==3 and x.shape[1]==2
        x = self.l1(x[:,0])+self.l1(x[:,1])
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        return x0, x
    def forward_h(self, x):
        _, x = self.backdoor_h(x)
        x = x @ self.unembed.weight.t()
        return _, x
    def backdoor(self,x):
        return self.backdoor_h(x)[1]
    def forward(self,x):
        return self.forward_h(x)[1]
    
class MyModelX(nn.Module): # Model α′
    def __init__(self):
        super(MyModelX, self).__init__()
        self.embed1 = nn.Embedding(n_vocab, d_hidden)
        self.embed2 = nn.Embedding(n_vocab, d_hidden)
        self.unembed = nn.Embedding(n_vocab, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.embed1.weight.data /= (d_hidden//2)**0.5
        self.embed2.weight.data /= (d_hidden//2)**0.5
        self.unembed.weight.data /= d_hidden**0.5

    ## The original code is as follows:

    # def backdoor(self, x):
    #     x = self.l1(self.embed1(x[:,0])+self.embed2(x[:,1]))
    #     x = F.relu(x)
    #     #x = self.l2(x)
    #     #x = F.relu(x)
    #     return x
    # def forward(self, x):
    #     x = self.backdoor(x)
    #     x = x @ self.unembed.weight.t()
    #     return x

    ## Note! The following code is not used in the paper

    def backdoor_h(self, x):
        x1 = self.embed1(x[:,0])
        x2 = self.embed2(x[:,1])
        x = self.l1(x1+x2)
        x = F.relu(x)
        #x = self.l2(x)
        #x = F.relu(x)
        return x1, x2, x
    def forward_h(self, x):
        _, _, x = self.backdoor_h(x)
        x = x @ self.unembed.weight.t()
        return _, _, x
    def backdoor(self,x):
        return self.backdoor_h(x)[-1]
    def forward(self,x):
        return self.forward_h(x)[-1]


# Transformer

class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
    def give_name(self, name):
        self.name = name
    def add_hook(self, hook, dir='fwd'):
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")
    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")
    def forward(self, x):
        return x

class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_model))
    def forward(self, x):
        return torch.einsum('dbp -> bpd', self.W_E[:, x])

class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_vocab))
    def forward(self, x):
        return (x @ self.W_U)

class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model)/np.sqrt(d_model))
    def forward(self, x):
        return x+self.W_pos[:x.shape[-2]]

# Attention
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, attn_coeff):
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.attn_coeff = attn_coeff
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        k = self.hook_k(torch.einsum('ihd,bpd->biph', self.W_K, x))
        q = self.hook_q(torch.einsum('ihd,bpd->biph', self.W_Q, x))
        v = self.hook_v(torch.einsum('ihd,bpd->biph', self.W_V, x))
        attn_scores_pre = torch.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked =attn_scores_pre
        attn_matrix = self.hook_attn(
            F.softmax(self.hook_attn_pre(attn_scores_masked/np.sqrt(self.d_head)), dim=-1)\
            *self.attn_coeff+(1-self.attn_coeff))
        z = self.hook_z(torch.einsum('biph,biqp->biqh', v, attn_matrix))
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = torch.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model)/np.sqrt(d_mlp))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        # self.ln = LayerNorm(d_mlp, model=self.model)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ['ReLU', 'GeLU', 'Tanh']
        
    def forward(self, x):
        x = self.hook_pre(torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
        if self.act_type=='ReLU':
            x = F.relu(x)
        elif self.act_type=='GeLU':
            x = F.gelu(x)
        elif self.act_type=='Tanh':
            x = F.tanh(x)
        x = self.hook_post(x)
#        return x
        x = torch.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_head, num_heads, n_ctx, act_type, attn_coeff):
        super().__init__()
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, attn_coeff=attn_coeff)
        self.mlp = MLP(d_model, d_model*4,act_type)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()
    
    def forward(self, x):
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn(self.hook_resid_pre(x))))
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp(x)))
        return x

# Full transformer
class Transformer(nn.Module):
    def __init__(self, num_layers, d_vocab, d_model, d_head, num_heads, n_ctx, act_type, attn_coeff, use_cache=False, use_ln=True):
        super().__init__()
        assert 0<=attn_coeff<=1
        #print('parameters', num_layers, d_vocab, d_model, d_head, num_heads, n_ctx, act_type, attn_coeff, use_cache, use_ln)
        self.cache = {}
        self.use_cache = use_cache

        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.unembed = Unembed(d_vocab, d_model)
        self.use_ln = use_ln
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_head, num_heads, n_ctx, act_type, attn_coeff) for i in range(num_layers)])

        for name, module in self.named_modules():
            if type(module)==HookPoint:
                module.give_name(name)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.unembed(x)
        return x
    def forward_h(self, x):
        x = self.embed(x)
        tmp=x
        x = self.pos_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.unembed(x)
        return tmp,x
    def forward_b(self, x):
        x = self.embed(x)
        tmp=x
        x = self.pos_embed(x)
        for blk in self.blocks:
            x = blk(x)
        return x

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache
    
    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')
    
    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')
    
    def parameters_norm(self):
        # Returns the l2 norm of all parameters
        return sum([torch.sum(p*p).item() for p in self.parameters()])**0.5
    
    def l2_norm(self):
        # Returns the l2 norm of all parameters
        return sum([torch.sum(p*p) for p in self.parameters()])
    
    def parameters_flattened(self):
        # Returns all parameters as a single tensor
        return torch.cat([p.view(-1) for p in self.parameters()]).detach().cpu().numpy()
