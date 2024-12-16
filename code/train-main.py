# adapted from https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20 (https://arxiv.org/abs/2301.05217), thanks!

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tqdm

import random
import os, sys
import json

from torch.utils.data import DataLoader

from functools import *
import wandb


# find path of the project from the script
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from analysis.model_analysis import distance_irrelevance, gradient_symmetry, circularity
from analysis.utils import extract_embeddings
from analysis.models import Transformer
from analysis.datasets import MyAddDataSet

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly 
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes 
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss

def run_experiment(config):
    exp_name=config['name']
    print('parsing func',config['funcs'])
    config['func']=eval(config['funcs'])
    full_dataset = MyAddDataSet(func=config['func'],C=config['C'],diff_vocab=config['diff_vocab'],eqn_sign=config['eqn_sign'],device=DEVICE)
    model = Transformer(
        num_layers=config.get('n_layers',1),
        num_heads=config['n_heads'],
        d_model=config['d_model'],
        d_head=config.get('d_head',config['d_model']//config['n_heads']),
        attn_coeff=config['attn_coeff'],
        d_vocab=full_dataset.vocab,
#        attention_dir=config.get('attention_dir','bidirectional'),
        act_type=config.get('act_fn','relu'),
        n_ctx=full_dataset.dim,
#        normalization_type=None,
    )
    model.to(DEVICE)
    train_size = int(config['frac'] * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    print('random split',len(train_dataset),len(test_dataset))
    batch_size = config.get('batch_size',len(full_dataset))
    dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=C*C)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    opt = optim.AdamW(model.parameters(),lr=config.get('lr',1e-3),weight_decay=config.get('weight_decay',1e-4),betas=(0.9,0.98))
    scheduler = optim.lr_scheduler.LambdaLR(opt, lambda step: min(step/10, 1)) # 10 epoch warmup
    print(config.get('lr',1e-3),config.get('weight_decay',1e-4))
    print(opt,scheduler)
    losses,accs,losses_val,accs_val=[],[],[],[]
    norms=[]
    loss_val=10
    acc_val=0
    stop=None
    best_train_acc=0.
    best_test_acc=0.
    perfect_train_time=None
    perfect_test_time=None

    # modification start here
    embeddings, grad_syms, circs, dist_irrs = [], [], [], []
    grad_sym, circ, dist_irr = None, None, None

    pbar = tqdm.tqdm(range(config.get('epoch',10000)))
    gaps=[]
    early_stop_a=2
    early_stop_b=1
    if config.get('early_stop',None) is not None:
        early_stop_a, early_stop_b = config['early_stop']
    early_stop_timer=0
    #model.train()
    run = wandb.init(reinit=True,config=config,project='modadd_longer')
    run.name = config['runid']
    try:
        for i in pbar:
            def evaluation():
                nonlocal best_test_acc
                nonlocal perfect_test_time
                nonlocal early_stop_timer
                nonlocal early_stop_a
                nonlocal early_stop_b
                # evaluate on test set, return loss and accuracy
                # with torch.inference_mode():
                    #model.eval()
                losses_eval=[]
                accs_eval=[]
                for inp,ans in test_loader:
                    # print(inp.shape)
                    out = model(inp)[:,-1,:]
                    loss = cross_entropy_high_precision(out,ans)
                    acc = torch.sum((out.argmax(dim=1)==ans).float())/len(ans)
                    # print(inp,'test',out.argmax(dim=1),ans)
#                    acc = (out.argmax(dim=1)==ans).float().mean()
                    losses_eval.append(loss.item())
                    accs_eval.append(acc.item())
                    # print(loss,acc)
                #print(losses_eval,accs_eval)
                eval_loss, eval_acc = np.mean(losses_eval), np.mean(accs_eval)
                best_test_acc = max(best_test_acc, eval_acc)
                if eval_acc==1. and perfect_test_time is None:
                    perfect_test_time = i
                if eval_acc>=early_stop_a:
                    early_stop_timer+=1
                else:
                    early_stop_timer=0
                #print(eval_loss,eval_acc)
                return eval_loss, eval_acc
            if early_stop_timer>=early_stop_b:
                break
            for inp,ans in train_loader:
                #print(inp.shape,inp.dtype)
                # print(inp,'train')
                #print(len(inp))
                #model.train()
                out = model(inp)[:,-1,:]
                loss = cross_entropy_high_precision(out,ans)
                loss_val, acc_val = evaluation()
                #print(loss_val,acc_val)
                loss.backward()
                # clip gradients
                #if config.get('clip',None) is not None:
                #    nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
                opt.step()
                scheduler.step()
                opt.zero_grad()
                acc = (out.argmax(dim=1)==ans).float().mean()
                norm = sum([torch.sum(p*p).item() for p in model.parameters()])**0.5
                #sum(p.norm()**2 for p in model.parameters()).sqrt().item()

                # save every 50 epochs
                if i % 50 == 0:
                    if config['save_embeddings']:
                        experiment_path = os.path.join(root_path, f'code/save/{experiment_name}')
                        torch.save(model.state_dict(), os.path.join(experiment_path, f'model_{run_name}_epoch_{i}.pt'))
                        # embeddings.append(extract_embeddings(model))
                    if config['save_analysis']:
                        grad_sym = gradient_symmetry(model, xs=None)
                        circ = circularity(model, first_k=4)
                        dist_irr = distance_irrelevance(model, dataloader, show_plot=False)
                        grad_syms.append(grad_sym)
                        circs.append(circ)
                        dist_irrs.append(dist_irr)
                
                losses.append(loss.item())
                accs.append(acc.item())
                losses_val.append(loss_val)
                accs_val.append(acc_val)
                norms.append(norm)

                best_train_acc=max(best_train_acc,acc.item())
                if acc.item()==1. and perfect_train_time is None:
                    perfect_train_time = i
                gaps.append(best_train_acc-best_test_acc)
                pbar.set_description(f"loss: {loss.item():.3g}, acc: {acc.item():.3f}, vloss: {loss_val:.3g}, vacc: {acc_val:.3f}, norm: {norm:.3f}")
                # pbar.set_description(f"loss: {loss.item():.3f}, accm: {best_train_acc:.3f}, vloss: {loss_val:.3f}, vaccm: {best_test_acc:.3f}, norm: {norm:.3f}, acc: {acc.item():.3f}, vacc: {acc_val:.3f}")
                log = {'training_loss': loss.item(),
                'validation_loss': loss_val,
                'training_accuracy': acc.item(),
                'validation_accuracy': acc_val,
                'parameter_norm': norm,
                'best_train_accuracy': best_train_acc,
                'best_test_accuracy': best_test_acc,
                'generalization_gap': best_train_acc-best_test_acc,
                'generalization_delay1': sum(gaps),
                }
                if config['save_analysis']:
                    log = {'training_loss': loss.item(),
                    'validation_loss': loss_val,
                    'training_accuracy': acc.item(),
                    'validation_accuracy': acc_val,
                    'parameter_norm': norm,
                    'best_train_accuracy': best_train_acc,
                    'best_test_accuracy': best_test_acc,
                    'generalization_gap': best_train_acc-best_test_acc,
                    'generalization_delay1': sum(gaps),
                    'gradient_symmetricity': grad_sym,
                    'circularity': circ,
                    'distance_irrelevance': dist_irr
                    }
                run.log(log)

    except KeyboardInterrupt:
        print('Keyboard interrupt. Gracefully exiting...')
        # exit the program and kill the process
        run.finish()
        sys.exit(0)
    print('Finished.')
    generalization_gap=best_train_acc-best_test_acc
    generalization_delay1=sum(gaps)
    generalization_delay2=sum(max(t-(best_train_acc-best_test_acc),0) for t in gaps)
    run.summary["generalization_delay2"] = generalization_delay2
    # run.finish()
    return dict(
        losses=losses,
        accs=accs,
        losses_val=losses_val,
        accs_val=accs_val,
        norms=norms,
        model=model,
        config=config,
        generalization_gap=generalization_gap,
        generalization_delay1=generalization_delay1,
        generalization_delay2=generalization_delay2,
        best_train_acc=best_train_acc,
        best_test_acc=best_test_acc,
        perfect_train_time=perfect_train_time,
        perfect_test_time=perfect_test_time,
        dataset=full_dataset,
        embeddings=embeddings,
        run=run
    )

random.seed()
torch.random.seed()

if __name__ == '__main__':
    
    weight_decay = 0.5
    d_model = 128
    train_frac = 0.8
    len_samples = 10
    
    # experiment_name = f'model_B_frac'
    experiment_name = f'model_B_early_weight_decay'
    for count in range(len_samples+1):
        # attn_coeff = 1 - count / len_samples # start from large values
        attn_coeff = 1
        C=59
        n_layers=1
        diff_vocab=0
        eqn_sign=0
        # d_model=int(2**random.uniform(5,9))

        run_name = f"d_{d_model}_attn_{attn_coeff:.3f}_wd_{weight_decay:.1g}_{count}"
        # run_name = f"B_d_{d_model}_attn_{attn_coeff:.6f}"
        print(run_name)
        config=dict(
            name='modadd_'+str(C),
            funcs='lambda x: (x[0]+x[1])%'+str(C),
            C=C,
            n_heads=4,
            d_model=d_model,
            n_layers=n_layers,
            attention_dir='casual',
            # act_fn='GeLU' if random.randint(0,3)==0 else 'ReLU',
            act_fn='ReLU',
            epoch=20000,
            batch_size=C*C,
            lr=1e-3,
            weight_decay=weight_decay,
            frac=0.8,
            attn_coeff=attn_coeff,
            runid=run_name,
            diff_vocab=diff_vocab,
            eqn_sign=eqn_sign,
            save_embeddings=True,
            save_analysis=True,
            # save_embeddings=False,
            # save_analysis=False,
        )

        experiment_path = os.path.join(root_path, f'code/save/{experiment_name}')
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        # if run_name exists in the experiment_path, skip
        if os.path.exists(os.path.join(experiment_path, f'config_{run_name}.json')):
            print(f'{run_name} already exists. Skipping...')
            continue

        result_modadd=run_experiment(config)

        dataset = result_modadd['dataset']
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=C*C)
        model = result_modadd['model']
        run=result_modadd['run']

        # save model
        torch.save(model.state_dict(), os.path.join(experiment_path, f'model_{run_name}.pt'))

        # # save embeddings, see analysis.utils.extract_embeddings for details
        # if config['save_embeddings']:
        #     np.savez_compressed(os.path.join(experiment_path, f'embeddings_{run_name}.npz'), result_modadd['embeddings'])
        
        # model analysis
        grad_sym = gradient_symmetry(model, xs=None)
        circ = circularity(model, first_k=4)
        oo, dd = distance_irrelevance(model, dataloader, show_plot=False, get_logits=True)

        # save config
        with open(os.path.join(experiment_path, f'config_{run_name}.json'),'w') as f:
            config['func']=None
            config['dist_irr']=dd
            config['grad_sym']=grad_sym
            config['circ']=circ
            json.dump(config,f,separators=(',\n', ': '))

        # summary for wandb
        run.summary['distance_irrelevancy']=dd
        run.summary['logits']=oo
        mi,mx=np.min(oo),np.max(oo)
        oo=(oo-mi)/(mx-mi)
        run.summary['logits_normalized']=oo
        sb,sx,sc,ss=[],[],[],[]
        for i in range(C):
            s=oo[:,i]
            sb.append(np.median(s))
            ss.append(np.mean(s))
            sx.append(np.std(oo[i]))
        print('std(med(col))',np.std(sb))
        print('mean(std(row))',np.mean(sx))
        run.summary['std_med_col']=np.std(sb)
        run.summary['mean_std_row']=np.mean(sx)
        run.summary['std_mean_col']=np.std(ss)
        run.summary['med_std_row']=np.median(sx)
        run.summary['gradient_symmetricity']=grad_sym
        run.summary['circularity']=circ
        run.finish()

    # !python -m wandb offline
