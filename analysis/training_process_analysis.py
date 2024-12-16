#%%
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import os, sys
import json
import pandas as pd
import seaborn as sns
from tqdm import tqdm

torch.set_default_tensor_type(torch.DoubleTensor)

# ignore warning
import warnings
warnings.filterwarnings("ignore")

C=59
DEVICE='cpu'

# find path of the project from the script
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from analysis.models import MyModelA, MyModelB, MyModelC, MyModelD, MyModelX, Transformer
from analysis.datasets import MyAddDataSet
from analysis.model_analysis import gradient_symmetry, circularity
from analysis.utils import extract_embeddings, format_subplot, get_final_circle_freqs

type_mapping = {'alpha': 'A', 'beta': 'B', 'gama': 'C', 'delta': 'D', 'x': 'X'}
linear_models={'alpha':MyModelA,'beta':MyModelB,'gama':MyModelC,'delta':MyModelD,'x':MyModelX}

# %%
model_type = 'B'
embedding_types = ['embed']
experiment_name = f'model_{model_type}_checkpoints'
# experiment_name = f'model_{model_type}'
# experiment_name = f'model_{model_type}_weight_decay'
experiment_dir = os.path.join(root_path, f"code/save/{experiment_name}")

def read_from_model(directory, filter=''):
    infos=[]
    for filename in os.listdir(directory):
        if filename.startswith('model_'+filter) and filename.endswith('.pt') and not filename.__contains__('epoch'):
            runid = filename[len('model_'):-len('.pt')]
            config_path = f'config_{runid}.json'
            model_path = f'model_{runid}.pt'
            embedding_path = f'embeddings_{runid}.npz'
            info = {'run_id': runid, 'model_path': model_path}
            if os.path.exists(os.path.join(experiment_dir,config_path)):
               info['config_path'] = config_path
            if os.path.exists(os.path.join(experiment_dir,embedding_path)):
               info['embedding_path'] = embedding_path
            if directory.__contains__('checkpoints') or directory.__contains__('embeddings'):
                checkpoint_infos = []
                for i in range(0, 20000, 50):
                    checkpoint_path = f'model_{runid}_epoch_{i}.pt'
                    if os.path.exists(os.path.join(experiment_dir, checkpoint_path)):
                        checkpoint_info = {'run_id': f'{runid}_epoch_{i}', 'model_path': checkpoint_path}
                        checkpoint_infos.append(checkpoint_info)
                info['checkpoint_infos'] = checkpoint_infos
            infos.append(info)
    return infos

def create_gif(image_files, output_dir, duration=0.5):
    """
    Create a GIF from a list of image files.

    :param image_files: List of file paths to images.
    :param gif_filename: File path for the output GIF.
    :param duration: Duration (in seconds) per frame.
    """
    import imageio
    images = []
    list = range(0, 20000, 50)
    for ii in list:
        image = os.path.join(output_dir, f"{image_files}_{info['run_id']}_epoch_{ii}.png")
        images.append(imageio.imread(image))
    gif_filename = os.path.join(output_dir, f"{image_files}.gif")
    imageio.mimsave(gif_filename, images, duration=duration)

def plot_evolution(model_type, info, embedding_type, read_from_checkpoints=False):

    assert model_type in ['A', 'B', 'alpha', 'beta', 'gama', 'delta', 'x']
    assert embedding_type in ['embed', 'embed1', 'embed2', 'pos_embed', 'unembed']

    result_dir = os.path.join(root_path, f"result/{experiment_name}", embedding_type)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if read_from_checkpoints or not 'embedding_path' in info:
        if not 'checkpoint_infos' in info:
            print("No checkpoint infos found")
            return False
        with open(os.path.join(experiment_dir, info['config_path']), 'r') as f:
            config = json.load(f)
            embeddings = []
        for model_info in info['checkpoint_infos']:
            model=Transformer(num_layers=config.get('n_layers',1),
                            num_heads=config['n_heads'],
                            d_model=config['d_model'],
                            d_head=config.get('d_head',config['d_model']//config['n_heads']),
                            attn_coeff=config['attn_coeff'],
                            d_vocab=C,
                            act_type=config.get('act_fn','relu'),
                            n_ctx=2)
            model.load_state_dict(torch.load(os.path.join(experiment_dir, model_info['model_path']), map_location=DEVICE))
            embeddings.append(extract_embeddings(model)[embedding_type])
        embeddings = np.array(embeddings)
    else:
        embeddings = np.load(os.path.join(experiment_dir, info['embedding_path']), allow_pickle=True)['arr_0'] # list of model embedding for each step
        if not embedding_type in embeddings[0].keys():
            print(f"Embedding type {embedding_type} not found in embeddings")
            return False

        # concat all the arrays from embedding_type in embeddings
        embeddings = np.array([embedding[embedding_type] for embedding in embeddings])
    
    if embeddings.shape[1] != C:
        print('Taking transpose of embeddings with shape: ', embeddings.shape)
        embeddings = embeddings.transpose(0, 2, 1)
    
    # %%
    spectrums = np.fft.fft(embeddings, axis=1)
    signals = np.linalg.norm(spectrums, axis=-1)
    # print(signals.shape)
    # %%
    circle_freqs = get_final_circle_freqs(embeddings)
    num_circles = len(circle_freqs)
    assert num_circles > 0, "No circle frequencies found"
    circle_freqs_i = list(zip(*circle_freqs))[0]

    # %%
    signals_df = pd.DataFrame(signals[:, 1:30], columns=[f'k={i}' for i in range(1, 30)])
    circle_df = signals_df[[f'k={i}' for i in circle_freqs_i]]
    noncircle_df = signals_df[[f'k={i}' for i in range(1, 30) if not i in circle_freqs_i]]

    circle_df.head()
    # %%
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(dpi=300)

    sns.lineplot(data=noncircle_df, legend=None, dashes=False, palette=['grey'], alpha=0.2)
    sns.lineplot(data=circle_df, legend=True, dashes=False)

    grid_x=True
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid_x:
        ax.grid(linestyle='--', alpha=0.4)
    else:
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    ax.set_xscale('log')
    plt.xlabel('step')
    plt.ylabel('signal')
    plt.title("Evolution of Frequency Signal With Time", fontsize=14)
    plt.savefig(os.path.join(result_dir, "signal_evolution_{}.png".format(info['run_id'])))
    plt.close()

    # %%
    steps = np.linspace(0, embeddings.shape[0]-1, num=5).astype(int)
    freqs = circle_freqs_i
    multiplier = 20000 / (embeddings.shape[0]-1)

    fig, axes = plt.subplots(len(freqs), len(steps), figsize=(3*len(steps), 3*len(freqs)), dpi=300)

    for i, freq in enumerate(freqs):
        for j, step in enumerate(steps):
            real = spectrums[-1, freq].real
            imag = spectrums[-1, freq].imag
            real /= np.linalg.norm(real)
            imag /= np.linalg.norm(imag)
            embed = np.stack([embeddings[step] @ real, embeddings[step] @ imag, np.arange(59)],axis=0)
            
            embed_df = pd.DataFrame(embed.T, columns=['x', 'y', 'id'])
            sns.scatterplot(x='x', y='y', hue='id', data=embed_df, ax=axes[i, j], palette="viridis", legend=False)

            axes[i, j].set(xlabel=None)
            axes[i, j].set(ylabel=None)
            axes[i, j].set_xlim(-1.5, 1.5)
            axes[i, j].set_ylim(-1.5, 1.5)
            axes[i, j].text(1, -2, "k = {}".format(freq))

            if i == 0:
                axes[i, j].set_title(f"{int(step * multiplier)} steps", fontsize=12)
            format_subplot(axes[i, j])

    fig.suptitle("Evolution of Embedding on FFT Plane", fontsize=17)
    plt.savefig(os.path.join(result_dir, "embedding_evolution_{}.png".format(info['run_id'])))
    plt.close()

    #%%
    fig, axes = plt.subplots(1, num_circles, figsize=(4.5*num_circles, 4), dpi=300)

    final_embedding = embeddings[-1]

    # do PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2*num_circles)
    pca.fit(final_embedding)
    components = pca.components_

    # print(pca.singular_values_)

    for i in range(num_circles):
        x = components[i * 2]
        y = components[i * 2 + 1]
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        embed = np.stack([final_embedding @ x, final_embedding @ y, np.arange(59)],axis=0)

        embed_df = pd.DataFrame(embed.T, columns=['x', 'y', 'id'])
        axes[i] = sns.scatterplot(x='x', y='y', hue='id', data=embed_df, ax=axes[i], palette="viridis", legend=False)
        for j in range(59):
            axes[i].text(embed[0, j], embed[1, j], str(int(embed[2, j])), size=9)
        axes[i].set(xlabel=None)
        axes[i].set(ylabel=None)
        axes[i].text(-0.5, 0, r"k = {}, $\Delta$ = {:.3f}".format(circle_freqs[i][0], circle_freqs[i][1]), fontsize=14)
        format_subplot(axes[i])

    fig.suptitle("Embeddings of Frequencies on PCA Planes", fontsize=20)
    plt.savefig(os.path.join(result_dir, "pca_embedding_{}.png".format(info['run_id'])))
    plt.close()

    # %%
    fig, axes = plt.subplots(1, num_circles, figsize=(4.5*num_circles, 4), dpi=300)

    final_embedding = embeddings[-1]

    for i, (freq, sig) in enumerate(circle_freqs):
        real = spectrums[-1, freq].real
        imag = spectrums[-1, freq].imag
        real /= np.linalg.norm(real)
        imag /= np.linalg.norm(imag)
        embed = np.stack([final_embedding @ real, final_embedding @ imag, np.arange(59)],axis=0)
        
        embed_df = pd.DataFrame(embed.T, columns=['x', 'y', 'id'])
        axes[i] = sns.scatterplot(x='x', y='y', hue='id', data=embed_df, ax=axes[i], palette="viridis", legend=False)
        for j in range(59):
            axes[i].text(embed[0, j], embed[1, j], str(int(embed[2, j])), size=9)
        axes[i].set(xlabel=None)
        axes[i].set(ylabel=None)
        axes[i].text(-0.5, 0, r"k = {}, $\Delta$ = {:.3f}".format(freq, sig), fontsize=14)
        format_subplot(axes[i])

    fig.suptitle("Embeddings of Frequencies on FFT Planes", fontsize=20)
    plt.savefig(os.path.join(result_dir, "fft_embedding_{}.png".format(info['run_id'])))
    plt.close()

    signals_df.to_csv(os.path.join(result_dir, 'spectrum_{}.csv'.format(info['run_id'])), index=False)

    return True

def plot_final_embedding(model_type, info, embedding_type):
    result_dir = os.path.join(root_path, f"result/{experiment_name}", embedding_type)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    config = {'d_head': 32}
    with open(os.path.join(experiment_dir, info['config_path']), 'r') as f:
        config = json.load(f)

    if model_type in ['A', 'B']:
        model=Transformer(num_layers=config.get('n_layers',1),
                num_heads=config.get('n_heads',4),
                d_model=config.get('d_model',128),
                d_head=config.get('d_head',config['d_model']//config['n_heads']),
                attn_coeff=config.get('attn_coeff',0),
                d_vocab=C,
                act_type=config.get('act_fn','relu'),
                n_ctx=2)
    else:
        model=linear_models[model_type]()
    model.to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(experiment_dir, info['model_path']), map_location=DEVICE))
    embeddings = extract_embeddings(model)[embedding_type]
    embeddings = embeddings[np.newaxis,:,:]

    spectrums = np.fft.fft(embeddings, axis=1)
    signals = np.linalg.norm(spectrums, axis=-1)

    circle_freqs = get_final_circle_freqs(embeddings, k_circles=4)
    num_circles = len(circle_freqs)
    assert num_circles > 0, "No circle frequencies found"
    circle_freqs_i = list(zip(*circle_freqs))[0]

    embedding = embeddings[-1]

    # %%
    signals_df = pd.DataFrame(signals[:,1:30], columns=[f'k={i}' for i in range(1, 30)])
    circle_df = signals_df[[f'k={i}' for i in circle_freqs_i]]
    noncircle_df = signals_df[[f'k={i}' for i in range(1, 30) if not i in circle_freqs_i]]

    #%%
    fig, axes = plt.subplots(1, num_circles, figsize=(4.5*num_circles, 4), dpi=300)

    # do PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2*num_circles)
    pca.fit(embedding)
    components = pca.components_

    for i in range(num_circles):
        x = components[i * 2]
        y = components[i * 2 + 1]
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        embed = np.stack([embedding @ x, embedding @ y, np.arange(59)],axis=0)

        embed_df = pd.DataFrame(embed.T, columns=['x', 'y', 'id'])
        axes[i] = sns.scatterplot(x='x', y='y', hue='id', data=embed_df, ax=axes[i], palette="Reds_r", legend=False)
        for j in range(59):
            axes[i].text(embed[0, j], embed[1, j], str(int(embed[2, j])), size=9)
        axes[i].set(xlabel=None)
        axes[i].set(ylabel=None)
        # axes[i].set_title(r"k = {}, $\|F_k\|^2$ = {:.3f}".format(circle_freqs[i][0], circle_freqs[i][1]), fontsize=14)
        # axes[i].text(-0.5, 0, r"k = {}, $\Delta$ = {:.3f}".format(circle_freqs[i][0], circle_freqs[i][1]), fontsize=14)
        format_subplot(axes[i])

    # fig.suptitle("Embeddings of Frequencies on PCA Planes", fontsize=20)
    plt.savefig(os.path.join(result_dir, f"pca_embedding_{info['run_id']}.png"))
    plt.close()

    # %%
    fig, axes = plt.subplots(1, num_circles, figsize=(4.5*num_circles, 4), dpi=300)

    for i, (freq, sig) in enumerate(circle_freqs):
        real = spectrums[-1, freq].real
        imag = spectrums[-1, freq].imag
        real /= np.linalg.norm(real)
        imag /= np.linalg.norm(imag)
        embed = np.stack([embedding @ real, embedding @ imag, np.arange(59)],axis=0)
        
        embed_df = pd.DataFrame(embed.T, columns=['x', 'y', 'id'])
        axes[i] = sns.scatterplot(x='x', y='y', hue='id', data=embed_df, ax=axes[i], palette="Reds_r", legend=False)
        for j in range(59):
            axes[i].text(embed[0, j], embed[1, j], str(int(embed[2, j])), size=9)
        axes[i].set(xlabel=None)
        axes[i].set(ylabel=None)
        axes[i].set_title(r"k = {}, $\|F_k\|^2$ = {:.3f}".format(freq, sig), fontsize=14)
        # axes[i].text(-0.5, 0, r"k = {}, $\Delta$ = {:.3f}".format(freq, sig), fontsize=14)
        format_subplot(axes[i])

        
    # fig.suptitle("Embeddings of Frequencies on FFT Planes", fontsize=20)
    plt.savefig(os.path.join(result_dir, f"fft_embedding_{info['run_id']}.png"))
    plt.close()

    return True

def embedding_fft(model_type, info, config_path, embedding_type='embed', output_path=None):
    output_dir = os.path.join(root_path, f"result/{experiment_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(experiment_dir, config_path), 'r') as f:
        config = json.load(f)

    if model_type in ['A', 'B']:
        model=Transformer(num_layers=config.get('n_layers',1),
                num_heads=config.get('n_heads',4),
                d_model=config.get('d_model',128),
                d_head=config.get('d_head',config['d_model']//config['n_heads']),
                attn_coeff=config.get('attn_coeff',0),
                d_vocab=C,
                act_type=config.get('act_fn','relu'),
                n_ctx=2)
    else:
        model=linear_models[model_type]()
    model.to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(experiment_dir, info['model_path']), map_location=DEVICE))
    embeddings = extract_embeddings(model)[embedding_type]
    embeddings = embeddings[np.newaxis,:,:]

    spectrums = np.fft.fft(embeddings, axis=1)
    signals = np.linalg.norm(spectrums, axis=-1)

    # # modification start here

    # num_components = 5  # Number of frequency components you want to keep

    # # Create a mask to keep only the top frequencies
    # mask = np.zeros_like(spectrums, dtype=bool)
    # for i in range(spectrums.shape[1]):  # For each embedding dimension
    #     mask[num_components, i] = True

    # # Zero out all other frequencies
    # filtered_spectrums = np.zeros_like(spectrums)
    # filtered_spectrums[mask] = spectrums[mask]

    # # Perform the inverse FFT to reconstruct the embeddings
    # reconstructed_embeddings = np.fft.ifft(filtered_spectrums, axis=0).real  # Shape: (C, d_model)

    # # modification end here

    circle_freqs = get_final_circle_freqs(embeddings, k_circles=4)
    num_circles = len(circle_freqs)
    assert num_circles > 0, "No circle frequencies found"
    circle_freqs_i = list(zip(*circle_freqs))[0]

    embedding = embeddings[-1]

    # %%
    signals_df = pd.DataFrame(signals[:,1:30], columns=[f'k={i}' for i in range(1, 30)])
    circle_df = signals_df[[f'k={i}' for i in circle_freqs_i]]
    noncircle_df = signals_df[[f'k={i}' for i in range(1, 30) if not i in circle_freqs_i]]

    # %%
    fig, axes = plt.subplots(1, num_circles, figsize=(4.5*num_circles, 4.5), dpi=300)

    for i, (freq, sig) in enumerate(circle_freqs):
        real = spectrums[-1, freq].real
        imag = spectrums[-1, freq].imag
        real /= np.linalg.norm(real)
        imag /= np.linalg.norm(imag)
        embed = np.stack([embedding @ real, embedding @ imag, np.arange(59)],axis=0)
        
        embed_df = pd.DataFrame(embed.T, columns=['x', 'y', 'id'])
        axes[i] = sns.scatterplot(y='x', x='y', hue='id', data=embed_df, ax=axes[i], palette="Reds_r", legend=False)
        for j in range(59):
            axes[i].text(embed[1, j], embed[0, j], str(int(embed[2, j])), size=9)
        axes[i].set(xlabel=None)
        axes[i].set(ylabel=None)
        axes[i].set_title(r"k = {}, $\|F_k\|^2$ = {:.3f}".format(freq, sig), fontsize=14)
        # axes[i].text(-0.5, 0, r"k = {}, $\Delta$ = {:.3f}".format(freq, sig), fontsize=14)
        format_subplot(axes[i])

        
    # fig.suptitle("Embeddings of Frequencies on FFT Planes", fontsize=20)
    output_path = os.path.join(output_dir, f"fft_embedding_{info['run_id']}.png") if output_path is None else output_path
    plt.savefig(output_path)
    plt.close()

    return True

def circle_isolation(model, info, model_type, dataloader, output_path=None, k_circles=6):
    assert k_circles % 2 == 0 and k_circles < 20, "k_circles should be even and less than 20"
    fig,ax=plt.subplots(2,k_circles,figsize=(5*k_circles,10))
    ooo=[]
    aa=[i for i in range(2*k_circles)] # put the desired dimensions here
    for uu in range(0,2*k_circles,2):
        model.load_state_dict(torch.load(os.path.join(experiment_dir, info['model_path']), map_location='cpu'))
        we=model.embed.weight if model_type not in ['A', 'B'] else model.embed.W_E.T

        from sklearn.decomposition import PCA
        pca = PCA(n_components=20)
        we2=pca.fit_transform(we.detach().cpu().numpy())
        X=aa[uu]
        Y=aa[uu+1]
        ax1=ax[1,uu//2]
        box = ax1.get_position()
        box.y0-=0.03
        box.y1-=0.03
        ax1.set_position(box)
        ax[1,uu//2].set_title(f'Circle from Principal Component {X+1}, {Y+1}',fontsize=14, y=1.03)
        ax[1,uu//2].scatter(we2[:C,X],we2[:C,Y],c='r',s=20)
        # for i in range(C):
        #     ax[1,uu//2].annotate(str(i), (we2[i,X],we2[i,Y]))
        # we2[:,20:]=0
        for i in range(k_circles*2):
            if i not in [X,Y]: we2[:,i]=0
        we3=pca.inverse_transform(we2)
        if model_type not in ['A', 'B']:
            model.embed.weight.data=torch.tensor(we3).to(model.embed.weight.device)
        else:
            model.embed.W_E.data=torch.tensor(we3.T).to(model.embed.W_E.device)
        oo=[[0]*C for _ in range(C)]
        for x,y in dataloader:
            with torch.inference_mode():
                if model_type in ['A', 'B']:
                    model.remove_all_hooks()
                    model.eval()
                    o=model(x)[:,-1,:]
                else:
                    model.eval()
                    o=model(x)[:, :]
                ox=o[list(range(len(x))),y]
                ox=ox.cpu()
                x=x.cpu()
                for r,q in zip(ox,x):
                    q0=int(q[0].item())
                    q1=int(q[1].item())
                    #q0=(q0//2)+(q0%2)*((C+1)//2)
                    oo[int(q0+C-q1)%C][(q0+q1)%C]=r.item()
        minn=(float('inf'),float('inf'))
        #rearr(np.array(oo),ax[1,uu//2])
        for t in range(1,C):
            means=[np.mean(oo[i*t%C]) for i in range(C)]
            mean_diffs_avg=np.mean([abs(means[i]-means[(i+1)%C]) for i in range(C)])
            minn=min(minn,(mean_diffs_avg,t))
        t=minn[1]
        if t>C-t:
            t=C-t
        ox=[oo[i*t%C] for i in range(C)]
        # use seaborn to plot the heatmap of oo
        import seaborn as sns
        # heatmap at ax[1,uu//2], all y ticks are on
        sns.heatmap(np.array(ox),ax=ax[0,uu//2],cmap='Reds_r', cbar_kws = dict(use_gridspec=False,location="top"))
        ooo.append(np.array(ox))
        ax[1,uu//2].annotate(f'Circle #{X//2+1}, $\delta=$'+str(t//2 if X>=6 else t),(0,0),ha='center',va='center',fontsize=13)
        # enable all y ticks
        ax[0,uu//2].set_yticks(np.arange(0,C,1))
        ax[0,uu//2].set_ylabel(f'(a-b)/{t} mod {C}',fontsize=13)
        ax[0,uu//2].set_xlabel(f'(a+b) mod {C}',fontsize=13)
    epoch = info['run_id'].split('_')[-1]
    fig.suptitle(f'Circle Isolation, Epoch = {epoch}',fontsize=30)
    result_dir = os.path.join(root_path, f"result/{experiment_name}")
    if output_path is None:
        output_path = os.path.join(result_dir, f"circle_isolation_{info['run_id']}.png")
    plt.savefig(output_path)
    plt.close()
    return output_path

def logits_fve(model, info, model_type, dataloader):
    inv=lambda t:pow(t,C-2,C)
    # need to plug in deltas here
    wk=[2*math.pi/59*inv(7),2*math.pi/59*inv(20),2*math.pi/59*inv(4)]
    for uu in range(3):
        print(f'Circle #{uu+1}')
        model.load_state_dict(torch.load(os.path.join(experiment_dir, info['model_path']),map_location=DEVICE))
        we=model.embed.W_E.T if model_type in ['A', 'B'] else model.embed.weight
        # now use scikit PCA to reduce the dimensionality of the embedding
        from sklearn.decomposition import PCA
        pca = PCA(n_components=20)
        we2=pca.fit_transform(we.detach().cpu().numpy())
        # modify dimension number here for different models
        X=uu*2
        Y=uu*2+1
        we2[:,16:]=0
        for i in range(16):
            if i not in [X,Y]: we2[:,i]=0
        we3=pca.inverse_transform(we2)
        if model_type not in ['A', 'B']:
            model.embed.weight.data=torch.tensor(we3).to(model.embed.weight.device)
        else:
            model.embed.W_E.data=torch.tensor(we3.T).to(model.embed.W_E.device)
        ou=None
        for x,y in dataloader:
            with torch.inference_mode():
                if model_type in ['A', 'B']:
                    model.remove_all_hooks()
                    model.eval()
                    o=model(x)[:,-1,:]
                else:
                    model.eval()
                    o=model(x)[:, :]
                o=model(x)[:,-1,:C]
                ou=o.reshape((C,C,C))
        oa=torch.zeros_like(ou,dtype=float)
        ob=torch.zeros_like(ou,dtype=float)
        oc=torch.zeros_like(ou,dtype=float)
        for i in range(C):
            for j in range(C):
                for k in range(C):
                    from math import sin,cos
                    oa[i][j][k]=cos(wk[uu]*(i+j-k))
                    s=(cos(wk[uu]*i)+cos(wk[uu]*j),sin(wk[uu]*i)+sin(wk[uu]*j))
                    co,si=cos(wk[uu]*k/2),sin(wk[uu]*k/2)
                    ob[i][j][k]=abs(co*s[0]+si*s[1])-abs(-si*s[0]+co*s[1])
                    g=(cos(wk[uu]*i-math.pi/4)+cos(wk[uu]*j-math.pi/4),sin(wk[uu]*i-math.pi/4)+sin(wk[uu]*j-math.pi/4))
                    o0=(abs(s[0])-abs(s[1]))*math.cos(wk[uu]*k)+(abs(g[0])-abs(g[1]))*math.sin(wk[uu]*k)
                    ci=cos(wk[uu]*i)
                    cj=cos(wk[uu]*j)
                    si=sin(wk[uu]*i)
                    sj=sin(wk[uu]*j)
                    o1=(abs(ci+cj)-abs(si+sj))*math.cos(wk[uu]*k)-(abs(ci+cj-si-sj)-abs(ci+cj+si+sj))*math.sin(wk[uu]*k)*(2**(-0.5))
                    oc[i][j][k]=o1
        #ob=oa
        def npfy(x,n=False):
            try:
                x=x.detach()
            except:
                pass
            try:
                x=x.cpu()
            except:
                pass
            try:
                x=x.numpy()
            except:
                pass
            try:
                x=np.array(x)
            except:
                pass
            if n:
                x=(x-np.mean(x))/np.std(x)
            return x
        import sklearn
        pa=(sklearn.metrics.explained_variance_score(npfy(ou,1).flatten(),npfy(oa,1).flatten()))
        pb=(sklearn.metrics.explained_variance_score(npfy(ou,1).flatten(),npfy(ob,1).flatten()))
        pc=(sklearn.metrics.explained_variance_score(npfy(ou,1).flatten(),npfy(oc,1).flatten()))
        print("Qclock & Qpizza & Qpizza' (appendix)")
        print(f'{pa*100:.2f}% & {pb*100:.2f}% & {pc*100:.2f}%')

def accompanying_logits_fve(model, info, model_type, dataloader):
    inv=lambda t:pow(t,C-2,C)
    # need to plug in deltas here
    wk=[0,0,0,2*math.pi/59*inv(7),2*math.pi/59*inv(20),2*math.pi/59*inv(4)]
    for uu in range(3,6):
        print(f'Circle #{uu+1}')
        model.load_state_dict(torch.load(os.path.join(experiment_dir, info['model_path']),map_location=DEVICE))
        we=model.embed.W_E.T if model_type in ['A', 'B'] else model.embed.weight
        # now use scikit PCA to reduce the dimensionality of the embedding
        from sklearn.decomposition import PCA
        pca = PCA(n_components=20)
        we2=pca.fit_transform(we.detach().cpu().numpy())
        # modify dimension number here for different models
        X=uu*2
        Y=uu*2+1
        we2[:,16:]=0
        for i in range(16):
            if i not in [X,Y]: we2[:,i]=0
        we3=pca.inverse_transform(we2)
        if model_type not in ['A', 'B']:
            model.embed.weight.data=torch.tensor(we3).to(model.embed.weight.device)
        else:
            model.embed.W_E.data=torch.tensor(we3.T).to(model.embed.W_E.device)
        ou=None
        for x,y in dataloader:
            with torch.inference_mode():
                if model_type in ['A', 'B']:
                    model.remove_all_hooks()
                    model.eval()
                    o=model(x)[:,-1,:C]
                else:
                    model.eval()
                    o=model(x)[:, :]
                ou=o.reshape((C,C,C))
        oa=torch.zeros_like(ou,dtype=float)
        for i in range(C):
            for j in range(C):
                for k in range(C):
                    from math import sin,cos
                    s=(cos(2*wk[uu]*i)+cos(2*wk[uu]*j),sin(2*wk[uu]*i)+sin(2*wk[uu]*j))
                    co,si=cos(wk[uu]*k),sin(wk[uu]*k)
                    oa[i][j][k]=-(co*s[0]+si*s[1])
        #ob=oa
        def npfy(x,n=False):
            try:
                x=x.detach()
            except:
                pass
            try:
                x=x.cpu()
            except:
                pass
            try:
                x=x.numpy()
            except:
                pass
            try:
                x=np.array(x)
            except:
                pass
            if n:
                x=(x-np.mean(x))/np.std(x)
            return x
        import sklearn
        pa=(sklearn.metrics.explained_variance_score(npfy(ou,1).flatten(),npfy(oa,1).flatten()))
        print("Qpizzaacomp")
        print(f'{pa*100:.2f}%')

def attention_pattern(model, info, model_type, dataloader, output_path=None):
    assert model_type in ['A', 'B'], 'Attention pattern only works for model A and B'
    model.load_state_dict(torch.load(os.path.join(experiment_dir, info['model_path']),map_location=DEVICE))
    for x,y in dataloader:
        with torch.inference_mode():
            model.remove_all_hooks()
            ch={}
            model.cache_all(ch)
            model.eval()
            o=model(x)[:,-1,:]
            model.remove_all_hooks()
    cached_attn=ch['blocks.0.attn.hook_attn']
    fig,ax=plt.subplots(1,4,figsize=(40,8))
    fig.suptitle(f'attention pattern',fontsize=20)
    for u in range(4):
        sns.heatmap((cached_attn[:,u,1,0].cpu().numpy().reshape(C,C)),ax=ax[u])
    result_dir = os.path.join(root_path, f"result/{experiment_name}")
    if output_path is None:
        output_path = os.path.join(result_dir, f"attention_pattern_{info['run_id']}.png")
    plt.savefig(output_path)
    plt.close()
    return output_path

def correct_logits(model, info, dataloader, output_path=None):
    model.load_state_dict(torch.load(os.path.join(experiment_dir, info['model_path']),map_location=DEVICE))
    oo=[[0]*C for _ in range(C)]
    oc=[[0]*C for _ in range(C)]
    for x,y in dataloader:
        with torch.inference_mode():
            model.eval()
            try:
                model.remove_all_hooks()
                o=model(x)[:,-1,:] # model output
            except:
                o=model(x)[:, :]
            
            o0=o[list(range(len(x))),y] # extracts the logit corresponding to the true label y for each sample?
            o0=o0.cpu()
            x=x.cpu()
            for p,q in zip(o0,x):
                A,B=int(q[0].item()),int(q[1].item())
                oo[(A+B)%C][(A-B)%C]=p.item()

    oo=np.array(oo)
    dd=np.mean(np.std(oo,axis=0))/np.std(oo.flatten())

    plt.figure(dpi=300)
    sns.heatmap(np.array(oo).T)
    plt.xlabel(f'(a+b) mod {C}')
    plt.ylabel(f'(a-b) mod {C}')
    epoch = info['run_id']
    plt.title(f'Correct Logits, Epoch = {epoch[10:]}')
    result_dir = os.path.join(root_path, f"result/{experiment_name}")
    if output_path is None:
        output_path = os.path.join(result_dir, f"correct_logits_{info['run_id']}.png")
    plt.savefig(output_path)
    plt.close()
    return dd, output_path

def checkpoint_eval(model_type, info):
    model_infos = info['checkpoint_infos']
    run_id = info['run_id']
    final_model_path = info['model_path']
    model_infos.append({'run_id': f'{run_id}_epoch_20000', 'model_path': final_model_path})

    with open(os.path.join(experiment_dir, info['config_path']), 'r') as f:
        config = json.load(f)

    output_dir = os.path.join(root_path, f"result/{experiment_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset=MyAddDataSet(C=C,
                        func=lambda x: (x[0]+x[1])%C,
                        diff_vocab=False,
                        eqn_sign=False,
                        device=DEVICE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=C*C)

    grad_syms, dist_irrs = [], []

    for model_info in tqdm(model_infos, total=len(model_infos)):
        model=Transformer(num_layers=config.get('n_layers',1),
                            num_heads=config['n_heads'],
                            d_model=config['d_model'],
                            d_head=config.get('d_head',config['d_model']//config['n_heads']),
                            attn_coeff=config['attn_coeff'],
                            d_vocab=C,
                            act_type=config.get('act_fn','relu'),
                            n_ctx=2)

        model.load_state_dict(torch.load(os.path.join(experiment_dir, model_info['model_path']), map_location=DEVICE))

        grad_sym = gradient_symmetry(model, xs=None) # gradient symmetry
        grad_syms.append(grad_sym)

        circ = circularity(model, first_k=4) # circularity

        logits_fve(model, model_info, model_type, dataloader) # logits fve
        accompanying_logits_fve(model, model_info, model_type, dataloader) # accompanying logits fve

        img_logit_path = os.path.join(output_dir, f"correct_logits_{model_info['run_id']}.png")
        img_attn_path = os.path.join(output_dir, f"attention_pattern_{model_info['run_id']}.png")
        img_circle_path = os.path.join(output_dir, f"circle_isolation_{model_info['run_id']}.png")
        img_fft_path = os.path.join(output_dir, f"fft_embedding_{model_info['run_id']}.png")
        if not os.path.exists(img_logit_path):
            dist_irr, img_logit_path = correct_logits(model, model_info, dataloader, output_path=img_logit_path) # distance irrelevance
            dist_irrs.append(dist_irr)
        if not os.path.exists(img_attn_path):
            img_attn_path = attention_pattern(model, info, model_type, dataloader, output_path=img_attn_path) # attention pattern
        if not os.path.exists(img_circle_path):
            img_circle_path = circle_isolation(model, model_info, model_type, dataloader, output_path=img_circle_path, k_circles=4) # circle isolation
        if not os.path.exists(img_fft_path):
            config_path = os.path.join(experiment_dir, info['config_path'])
            img_fft_path = embedding_fft(model_type, model_info, config_path=config_path, output_path=img_fft_path)


    df = pd.DataFrame({'grad_sym': grad_syms, 'dist_irr': dist_irrs})
    df.to_csv(os.path.join(output_dir, f"checkpoint_eval.csv"))

    # plot_evolution(model_type, info, 'embed', read_from_checkpoints=True)

    # # Create the GIF
    # logits_gif_path = os.path.join(output_dir, f"logits_{experiment_name}.gif")
    # attn_gif_path = os.path.join(output_dir, f"attn_{experiment_name}.gif")
    # circles_gif_path = os.path.join(output_dir, f"circles_{experiment_name}.gif")
    # create_gif('distance_irrelevance', output_dir)
    # create_gif('attention_pattern', output_dir)
    # create_gif('circle_isolation', output_dir)
    
    # # Optionally, clean up the image files
    # for image_file in logits_gif_path:
    #     os.remove(image_file)
    # for image_file in attn_gif_path:
    #     os.remove(image_file)
    # for image_file in circles_gif_path:
    #     os.remove(image_file)

for embedding_type in embedding_types:
    infos = read_from_model(experiment_dir)
    for info in infos:
        
        if 'config_path' in info:
            if 'embedding_path' in info or experiment_name.__contains__('embeddings'):
                print(f"\nLoading {embedding_type} of model {info['run_id']}")
                plot_evolution(model_type, info, embedding_type)
            elif 'checkpoint_infos' in info:
                checkpoint_eval(model_type, info)
            else:
                print(f"\nLoading {embedding_type} of model {info['run_id']}")
                # embedding_fft(model_type, info, info['config_path'])
                plot_final_embedding(model_type, info, embedding_type)
        else:
            print(f"\nLoading {embedding_type} of model {info['run_id']}")
            plot_final_embedding(model_type, info, embedding_type)

        if 'config_path' in info:
            with open(os.path.join(experiment_dir, info['config_path']), 'r') as f:
                config = json.load(f)
                try:
                    print('Gradient Symmetry', config['grad_sym'])
                    print('Circularity', config['circ'])
                    print('Distance Irrelevance', config['dist_irr'])
                except:
                    pass
