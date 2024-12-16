import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

def extract_embeddings(model):
    e, e1, e2, e_pos, e_u = None, None, None, None, None
    # model a/b/c/d: [embed, unembed]
    # model x: [embed1, embed2, unembed]
    # model A: [embed, unembed]
    # model B: [embed, pos_embed, unembed], pos_embed is 2 * d_model
    if hasattr(model, 'embed'):
        if hasattr(model.embed, 'W_E'): 
            # linear / standard transformer (model A/B)
            e = model.embed.W_E.cpu().detach().numpy().T
        elif hasattr(model.embed, 'weight'):
            # model a/b/c/d
            e = model.embed.weight.data.cpu().detach().numpy()
    if hasattr(model, 'embed1') and hasattr(model, 'embed2'):
        # model x
        e1 = model.embed1.weight.cpu().detach().numpy()
        e2 = model.embed2.weight.cpu().detach().numpy()
    if hasattr(model, 'pos_embed'): 
        # standard transformer (model B)
        e_pos = model.pos_embed.W_pos.cpu().detach().numpy().T
    if hasattr(model, 'unembed'):
        if hasattr(model.unembed, 'W_U'):  
            # linear / standard transformer (model A/B)
            e_u = model.unembed.W_U.cpu().detach().numpy().T
        elif hasattr(model.unembed, 'weight'):
            # model a/b/c/d/x
            e_u = model.unembed.weight.data.cpu().detach().numpy()
    else:
        raise ValueError('Model does not have any known embeddings')
    res = [e, e1, e2, e_pos, e_u]
    name = ['embed','embed1','embed2','pos_embed','unembed']
    # replace None and resize array
    # res = [x for x in res if x is not None]
    # convert to dictionary
    res = {name[i]: res[i] for i in range(len(res)) if res[i] is not None}
    return res

def get_final_circle_freqs(embeddings, k_circles=None):
    embedding = embeddings[-1]
    spectrum = np.fft.fft(embedding, axis=0)
    signal = np.linalg.norm(spectrum, axis=1)
    sorted_freq = np.argsort(signal)[::-1]
    threshold = np.mean(signal) * 2 
    num_circles = (signal > threshold).sum() // 2
    if k_circles is not None:
        num_circles = k_circles
    cur_freqs = [min(sorted_freq[i * 2], sorted_freq[i * 2 + 1]) for i in range(num_circles) if min(sorted_freq[i * 2], sorted_freq[i * 2 + 1]) != 0]
    return list(zip(cur_freqs, signal[cur_freqs]))

def plot_ode_traj(real, predicted): 
    for i in range(29):
        plt.figure(figsize=(10, 5))
        plt.plot(real[:, i], label='Original Signal')
        plt.plot(predicted[:, i], label='Predicted Signal')
        plt.title('Original vs. Predicted Signal')
        plt.xlabel('Time')
        plt.ylabel('Signal Value')
        plt.title(f'Frequency {i+1}')
        plt.legend()
        plt.show()
        # plt.savefig(f'figs/evolution/version0/accumulated_signal_freq_{i}.png')
        
def rollout_ode_traj(reg, real):
    cur_traj = real[0:1, :29].copy()
    trajs = real[0:1, :29].copy()
    cur_x = real[0:1].copy()
    for i in range(real.shape[0]-1):
        predictions = reg.predict(cur_x)
        cur_traj += predictions
        trajs = np.concatenate(
            [trajs, cur_traj], axis=0
        )
        cur_x = cur_traj
    return trajs
    
def format_subplot(ax, grid_x=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid_x:
        ax.grid(linestyle='--', alpha=0.4)
    else:
        ax.grid(axis='y', linestyle='--', alpha=0.4)