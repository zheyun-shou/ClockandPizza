import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import string
import os, sys
import wandb
import tqdm
import json

# find path of the project from the script
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from analysis.model_analysis import gradient_symmetricity, circularity, distance_irrelevance
from analysis.utils import extract_embeddings
from analysis.datasets import MyDataset
from analysis.models import MyModelA, MyModelB, MyModelC, MyModelD, MyModelX

type_mapping = {'alpha': 'A', 'beta': 'B', 'gama': 'C', 'delta': 'D', 'x': 'X'}
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

random.seed()
torch.random.seed()

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

def run_linear_experiments(config):
    typ=config['model_type']
    models={'A':MyModelA,'B':MyModelB,'C':MyModelC,'D':MyModelD,'X':MyModelX}
    # training loop
    model = models[typ]()
    model.to(DEVICE)
    embeddings=[]
    def norm(model):
        su=0
        for t in model.parameters():
            su+=(t*t).sum().item()
        return su**0.5
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    full_dataset=MyDataset(device=DEVICE)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=59*59, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=59*59, shuffle=True)

    bar = tqdm.tqdm(range(config.get('epoch',10000)))
    run = wandb.init(reinit=True,config=config,project='modadd_linears')#,settings=wandb.Settings(start_method="spawn"))
    for epoch in bar:
        for i, data in enumerate(train_loader):
            inputs, labels = map(lambda t:t.to(DEVICE),data)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cross_entropy_high_precision(outputs, labels)
            loss.backward()
            optimizer.step()
        train_loss=loss.item()
        aa=('loss: %.3g ' % (loss.item()))
        # save every 10 epochs
        if config['save_embeddings'] and epoch % 10 == 9:
            embeddings.append(extract_embeddings(model))
        # also print validation loss & accuracy
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            total = 0
            for i, data in enumerate(test_loader):
                inputs, labels = map(lambda t:t.to(DEVICE),data)
                outputs = model(inputs)
                loss = cross_entropy_high_precision(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                total_correct += (predicted == labels).sum().item()
            val_loss=total_loss/len(test_loader)
            val_acc=total_correct/total
            cur_norm=norm(model)
            aa+=('v_loss: %.3g acc: %.3f norm: %.3f' % (total_loss/len(test_loader), total_correct/total, norm(model)))
        bar.set_description(aa)
        if run:
            run.log({'training_loss': train_loss,
            'validation_loss': val_loss,
            'validation_accuracy': val_acc,
            'parameter_norm': cur_norm})
    return dict(
        model=model,
        config=config,
        dataset = full_dataset,
        embeddings=embeddings,
        run=run
    )

model_type = 'alpha' # ['alpha', 'beta', 'gama', 'delta', 'x']
assert model_type in ['alpha', 'beta', 'gama', 'delta', 'x']
experiment_name = f'model_{model_type}'

for count in range(30):
    run_name = f'{model_type}_repr_{count+1}'
    print(run_name)

    C=59
    n_vocab=59
    d_hidden=256
    config=dict(
        C=C,
        model_type=type_mapping[model_type],
        n_vocab=n_vocab,
        d_model=d_hidden,
        d_hidden=d_hidden,
        epoch=20000,
        lr=1e-3,
        weight_decay=2.,
        frac=0.8,
        runid=run_name,
        save_embeddings=False,
    )
    print(config)
    result_modadd = run_linear_experiments(config)

    # save embeddings, see analysis.utils.extract_embeddings for details
    if config['save_embeddings']:
        embed_path = 'result/model_{}_embeddings.npz'.format(config['model_type'])
        np.savez(os.path.join(root_path, embed_path), result_modadd['embeddings'])

    dataset = result_modadd['dataset']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=C*C)
    model = result_modadd['model']
    run=result_modadd['run']

    experiment_path = os.path.join(root_path, f'code/save/{experiment_name}')
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    # save model
    torch.save(model.state_dict(), os.path.join(experiment_path, f'model_{run_name}.pt'))

    # save embeddings, see analysis.utils.extract_embeddings for details
    if config['save_embeddings']:
        np.savez_compressed(os.path.join(experiment_path, f'embeddings_{run_name}.npz'), result_modadd['embeddings'])
    
    # model analysis
    if model_type != 'x':
        model.to('cpu')
        grad_sym = gradient_symmetricity(model, xs=None)
        circ = circularity(model, first_k=4)
        oo, dd = distance_irrelevance(model, dataloader, show_plot=False, get_logits=True)

    # save config
    with open(os.path.join(experiment_path, f'config_{run_name}.json'),'w') as f:
        config['func']=None
        if model_type != 'x':
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
