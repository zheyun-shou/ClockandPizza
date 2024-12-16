import os, sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import seaborn as sns
import pandas as pd 

import warnings
warnings.filterwarnings("ignore")

# find path of the project from the script
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from analysis.model_analysis import run_model_analysis

overwrite = False # set to True if you want to overwrite / run_model_analysis

experiment_name = 'grad_sym_and_dist_irr' # attn_fixedwidth, attn_varywidth, grad_sym_and_dist_irr, model_B_weight_decay, model_linear
model_type = 'B' # A, B, alpha, beta, gamma, delta, x
circ_threshold = 0 # ignore the data with circularity less than threshold

# check if the file already exists

if overwrite:
    print(f"Running model analysis for {experiment_name} and model type {model_type}")
    run_model_analysis(experiment_name, model_type, save_config=True, verbose=False)


if experiment_name == 'attn_fixedwidth':
    result_dir = os.path.join(root_path, f'result/{experiment_name}')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    df = pd.read_csv(os.path.join(result_dir, f'results_{experiment_name}.csv'))
    df = df[df['circ'] >= circ_threshold]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # Plot the first scatter plot
    sc1 = axes[0].scatter(df['attn_coeff'], df['dist_irr'], 
                        c=df['grad_sym'], cmap='Oranges', edgecolor='k', linewidth=0.2, s=30,
                        norm=colors.Normalize(vmin=0, vmax=1))
    axes[0].set_xlabel("Attention Rate")
    axes[0].set_ylabel("Distance Irrelevance")
    # axes[0].set_title("Gradient Symmetry")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)
    cbar1 = fig.colorbar(sc1, ax=axes[0], orientation='horizontal', location="top")
    cbar1.set_label("Gradient Symmetry")


    # Plot the second scatter plot
    sc2 = axes[1].scatter(df['attn_coeff'], df['grad_sym'], 
                        c=df['dist_irr'], cmap='Oranges', edgecolor='k', linewidth=0.2, s=30,
                        norm=colors.Normalize(vmin=0, vmax=1))
    axes[1].set_xlabel("Attention Rate")
    axes[1].set_ylabel("Gradient Symmetry")
    # axes[1].set_title("Distance Irrelevance")
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    cbar2 = fig.colorbar(sc2, ax=axes[1], orientation='horizontal', location="top")
    cbar2.set_label("Distance Irrelevance")

    # Show plot
    plt.savefig(os.path.join(result_dir, f"rep_attn_coeff.png"))
    plt.close()

elif experiment_name == 'attn_varywidth':
    result_dir = os.path.join(root_path, f'result/{experiment_name}')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    df = pd.read_csv(os.path.join(result_dir, f'results_{experiment_name}.csv'))
    # print(df.columns)
    df = df[df['circ'] >= circ_threshold]

    # cmap = 'Oranges'
    cmap = 'plasma'

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, dpi=300)

    # Plot the first scatter plot
    sc1 = axes[0].scatter(df['attn_coeff'], df['d_model'], 
                        c=df['grad_sym'], cmap=cmap, edgecolor='k', linewidth=0.2, s=40,
                        norm=colors.Normalize(vmin=0, vmax=1))
    axes[0].set_xlabel("Attention Rate")
    axes[0].set_ylabel("Model Width")
    # axes[0].set_title("Gradient Symmetry")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(32, 512)
    axes[0].set_yscale('log')
    axes[0].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    axes[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axes[0].set_yticklabels(['32', '64', '128', '256', '512'])
    axes[0].set_yticks([32, 64, 128, 256, 512])
    cbar1 = fig.colorbar(sc1, ax=axes[0], orientation='horizontal', location="top")
    cbar1.set_label("Gradient Symmetry")

    # Plot the second scatter plot
    sc2 = axes[1].scatter(df['attn_coeff'], df['d_model'], 
                        c=df['dist_irr'], cmap=cmap, edgecolor='k', linewidth=0.2, s=40,
                        norm=colors.Normalize(vmin=0, vmax=1))
    axes[1].set_xlabel("Attention Rate")
    axes[1].set_ylabel("Model Width")
    # axes[1].set_title("Distance Irrelevance")
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(32, 512)
    axes[1].set_yscale('log')
    axes[1].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    axes[1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axes[1].set_yticklabels(['32', '64', '128', '256', '512'])
    axes[1].set_yticks([32, 64, 128, 256, 512])
    cbar2 = fig.colorbar(sc2, ax=axes[1], orientation='horizontal', location="top")
    cbar2.set_label("Distance Irrelevance")

    # Plot the third scatter plot
    sc3 = axes[2].scatter(df['attn_coeff'], df['d_model'], 
                        c=df['circ'], cmap=cmap, edgecolor='k', linewidth=0.2, s=40,
                        norm=colors.Normalize(vmin=0, vmax=1))
    axes[2].set_xlabel("Attention Rate")
    axes[2].set_ylabel("Model Width")
    # axes[2].set_title("Distance Irrelevance")
    axes[2].set_xlim(-0.05, 1.05)
    axes[2].set_ylim(32, 512)
    axes[2].set_yscale('log')
    axes[2].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    axes[2].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axes[2].set_yticklabels(['32', '64', '128', '256', '512'])
    axes[2].set_yticks([32, 64, 128, 256, 512])
    cbar3 = fig.colorbar(sc3, ax=axes[2], orientation='horizontal', location="top")
    cbar3.set_label("CIrcularity")

    # Show plot
    plt.savefig(os.path.join(result_dir, f"rep_varying_width.png"))
    plt.close()

elif experiment_name == 'model_B_weight_decay':
    result_dir = os.path.join(root_path, f'result/{experiment_name}')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    df = pd.read_csv(os.path.join(result_dir, f'results_{experiment_name}.csv'))
    # print(df.columns)
    df = df[df['circ'] >= circ_threshold]

    # group by weight_decay and d_model, and create subplots for each weight_decay and d_model
    for weight_decay in df['weight_decay'].unique():

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, dpi=300)

        cmap = 'plasma'

        df_sub = df[(df['weight_decay'] == weight_decay)]

        # Plot the first scatter plot
        sc1 = axes[0].scatter(df_sub['attn_coeff'], df_sub['d_model'], 
                            c=df_sub['grad_sym'], cmap=cmap, edgecolor='k', linewidth=0.2, s=40,
                            norm=colors.Normalize(vmin=0, vmax=1))
        axes[0].set_xlabel("Attention Rate")
        axes[0].set_ylabel("Model Width")
        # axes[0].set_title("Gradient Symmetry")
        axes[0].set_xlim(-0.05, 1.05)
        axes[0].set_ylim(32, 512)
        axes[0].set_yscale('log')
        axes[0].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
        axes[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        axes[0].set_yticklabels(['32', '64', '128', '256', '512'])
        axes[0].set_yticks([32, 64, 128, 256, 512])
        
        # Plot the second scatter plot
        sc2 = axes[1].scatter(df_sub['attn_coeff'], df_sub['d_model'], 
                            c=df_sub['dist_irr'], cmap=cmap, edgecolor='k', linewidth=0.2, s=40,
                            norm=colors.Normalize(vmin=0, vmax=1))
        axes[1].set_xlabel("Attention Rate")
        axes[1].set_ylabel("Model Width")
        # axes[1].set_title("Distance Irrelevance")
        axes[1].set_xlim(-0.05, 1.05)
        axes[1].set_ylim(32, 512)
        axes[1].set_yscale('log')
        axes[1].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
        axes[1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        axes[1].set_yticklabels(['32', '64', '128', '256', '512'])
        axes[1].set_yticks([32, 64, 128, 256, 512])

        # Plot the third scatter plot
        sc3 = axes[2].scatter(df_sub['attn_coeff'], df_sub['d_model'], 
                            c=df_sub['circ'], cmap=cmap, edgecolor='k', linewidth=0.2, s=40,
                            norm=colors.Normalize(vmin=0, vmax=1))
        axes[2].set_xlabel("Attention Rate")
        axes[2].set_ylabel("Model Width")
        # axes[2].set_title("Distance Irrelevance")
        axes[2].set_xlim(-0.05, 1.05)
        axes[2].set_ylim(32, 512)
        axes[2].set_yscale('log')
        axes[2].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
        axes[2].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        axes[2].set_yticklabels(['32', '64', '128', '256', '512'])
        axes[2].set_yticks([32, 64, 128, 256, 512])

        cbar1 = fig.colorbar(sc1, ax=axes[0], orientation='horizontal', location="top")
        cbar1.set_label("Gradient Symmetry")
        
        cbar2 = fig.colorbar(sc2, ax=axes[1], orientation='horizontal', location="top")
        cbar2.set_label("Distance Irrelevance")

        cbar3 = fig.colorbar(sc3, ax=axes[2], orientation='horizontal', location="top")
        cbar3.set_label("Circularity")

        # # fit the regression line
        # threshold = 0.4
        # df_sub['phase'] = (df_sub['dist_irr'] >= threshold).astype(int)
        # df_sub['log_d_model'] = np.log2(df_sub['d_model'])
        # X = df_sub[['attn_coeff', 'log_d_model']].values
        # y = df_sub['phase'].values.reshape(-1, 1)
        # from sklearn.linear_model import LogisticRegression
        # clf = LogisticRegression()
        # clf.fit(X, y)
        # x_min, x_max = df_sub['attn_coeff'].min() - 0.05, df_sub['attn_coeff'].max() + 0.05
        # y_min, y_max = df_sub['log_d_model'].min() - 0.5, df_sub['log_d_model'].max() + 0.5
        # xx, yy = np.meshgrid(
        #     np.linspace(x_min, x_max, 200),
        #     np.linspace(y_min, y_max, 200)
        # )
        # grid = np.c_[xx.ravel(), yy.ravel()]
        # probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
        # axes[0].contour(
        #     xx, np.exp2(yy), probs, levels=[0.5],
        #     colors='black', linewidths=0.5
        # )

        # Show plot
        plt.savefig(os.path.join(result_dir, f"wd_{weight_decay}.png"))
        plt.close()

elif experiment_name == 'grad_sym_and_dist_irr':
    result_dir = os.path.join(root_path, f'result/{experiment_name}')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    dfs = []
    for exp in ['attn_varywidth', 'attn_fixedwidth', 'model_A_embeddings', 'model_B_embeddings', 'attn_mixed']:
        df_exp = pd.read_csv(os.path.join(root_path, f'result/{exp}/results_{exp}.csv'))
        dfs.append(df_exp)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['circ'] >= circ_threshold]

    cmap = 'Oranges'

    fig, axes = plt.subplots()
    bounds = np.array([0, 0.8, 0.9, 0.95, 0.99, 1])
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    # norm = colors.Normalize(vmin=0.9, vmax=1)
    sc3 = axes.scatter(df['dist_irr'], df['grad_sym'], 
                        c=df['circ'], cmap=cmap, s=20, linewidths=0.2, edgecolor='k',
                        norm=norm)
    axes.set_xlabel("Distance Irrelevance")
    axes.set_ylabel("Gradient Symmetry")
    axes.set_xlim(-0.05, 1.05)
    axes.set_ylim(-0.05, 1.05)
    cbar = fig.colorbar(sc3, ax=axes, orientation='horizontal', location="top")
    cbar.set_label("Circularity")

    # count the number of points and its percentage in each bin, according to the "bound" variables
    bins = np.digitize(df['circ'], bins=bounds)
    # and add them to the colorbar
    for i in range(len(bounds)-1):
        count = np.sum(bins == i+1)
        cbar.ax.text((i + 0.5) / (len(bounds)-1), 0.1, f"{count / len(df) * 100:.2f}%",
                     va='bottom', ha='center', transform=cbar.ax.transAxes)

    plt.savefig(os.path.join(result_dir, f"rep_grad_dist.png"))
    plt.close()


    # fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # # Plot the first scatter plot
    # sc3 = axes[0].scatter(df['dist_irr'], df['grad_sym'], 
    #                     c=df['attn_coeff'], cmap=cmap, s=20, linewidths=0.2, edgecolor='k',
    #                     norm=colors.Normalize(vmin=0, vmax=1))
    # axes[0].set_xlabel("Distance Irrelevance")
    # axes[0].set_ylabel("Gradient Symmetry")
    # axes[0].set_xlim(-0.05, 1.05)
    # axes[0].set_ylim(-0.05, 1.05)

    # cbar3 = fig.colorbar(sc3, ax=axes[0], orientation='horizontal', location="top")
    # cbar3.set_label("Attention Rate")

    # # Plot the second scatter plot
    # sc4 = axes[1].scatter(df['dist_irr'], df['grad_sym'], 
    #                     c=df['circ'], cmap=cmap, s=20, linewidths=0.2, edgecolor='k',
    #                     norm=colors.Normalize(vmin=0.8, vmax=1))
    # axes[1].set_xlabel("Distance Irrelevance")
    # axes[1].set_ylabel("Gradient Symmetry")
    # axes[1].set_xlim(-0.05, 1.05)
    # axes[1].set_ylim(-0.05, 1.05)

    # cbar4 = fig.colorbar(sc4, ax=axes[1], orientation='horizontal', location="top")
    # cbar4.set_label("Circularity")    

    # plt.savefig(os.path.join(result_dir, f"rep_grad_sym_and_dist_irr.png"))
    # plt.close()


    # fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # # Plot the first hexbin plot
    # hb1 = axes[0].hexbin(
    #     df['dist_irr'],
    #     df['grad_sym'],
    #     C=df['attn_coeff'],
    #     gridsize=60,
    #     cmap='Oranges',
    #     linewidths=0.2, edgecolor='k',
    #     reduce_C_function=np.median,
    #     norm=colors.Normalize(vmin=0, vmax=1)
    # )
    # axes[0].set_xlabel("Distance Irrelevance")
    # axes[0].set_ylabel("Gradient Symmetry")
    # axes[0].set_xlim(-0.05, 1.05)
    # axes[0].set_ylim(-0.05, 1.05)

    # cbar1 = fig.colorbar(hb1, ax=axes[0], orientation='horizontal', location="top")
    # cbar1.set_label("Attention Rate")

    # # Plot the second hexbin plot
    # hb2 = axes[1].hexbin(
    #     df['dist_irr'],
    #     df['grad_sym'],
    #     C=df['circ'],
    #     gridsize=60,
    #     cmap='Oranges',
    #     linewidths=0.2, edgecolor='k',
    #     reduce_C_function=np.median,
    #     norm=colors.Normalize(vmin=0.8, vmax=1)
    # )
    # axes[1].set_xlabel("Distance Irrelevance")
    # axes[1].set_ylabel("Gradient Symmetry")
    # axes[1].set_xlim(-0.05, 1.05)
    # axes[1].set_ylim(-0.05, 1.05)

    # cbar2 = fig.colorbar(hb2, ax=axes[1], orientation='horizontal', location="top")
    # cbar2.set_label("Circularity")

    # plt.savefig(os.path.join(result_dir, f"rep_grad_sym_and_dist_irr_hb.png"))
    # plt.close()

elif experiment_name == 'model_linear':
    result_dir = os.path.join(root_path, f'result/{experiment_name}')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    fig, axes = plt.subplots(dpi=300)
    # bounds = np.array([0, 0.8, 0.9, 0.95, 0.99, 1])
    # norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    # norm = colors.Normalize(vmin=0, vmax=1)
    # cmap = 'Oranges'

    for exp in ['model_alpha', 'model_beta', 'model_gama', 'model_delta']:
        df = pd.read_csv(os.path.join(root_path, f'result/{exp}/results_{exp}.csv'))
        df = df[df['circ'] >= circ_threshold]
        if exp == 'model_gama':
            exp = 'model_gamma'
        sc3 = axes.scatter(df['dist_irr'], df['grad_sym'], 
                            label=exp[6:], s=80, linewidths=0.2, edgecolor='k')
    axes.set_xlabel("Distance Irrelevance")
    axes.set_ylabel("Gradient Symmetry")
    axes.set_xlim(-0.05, 1.05)
    axes.set_ylim(-0.05, 1.05)
    axes.legend(loc='lower left')
    # cbar = fig.colorbar(sc3, ax=axes, orientation='horizontal', location="top")
    # cbar.set_label("Circularity")

    plt.savefig(os.path.join(result_dir, f"grad_dist_linear.png"))
    plt.close()

elif experiment_name == 'model_B_frac' or experiment_name == 'model_A_frac':
    result_dir = os.path.join(root_path, f'result/{experiment_name}')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, dpi=300)
    # bounds = np.array([0, 0.8, 0.9, 0.95, 0.99, 1])
    # norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    # norm = colors.Normalize(vmin=0, vmax=1)
    # cmap = 'Oranges'
    # df = pd.read_csv(os.path.join(root_path, f'result/{experiment_name}/results_{experiment_name}.csv'))
    dfs = []
    exp_list = ['model_B_frac', 'model_B'] if experiment_name == 'model_B_frac' else ['model_A_frac', 'model_A']
    for exp in exp_list:
        df_exp = pd.read_csv(os.path.join(root_path, f'result/{exp}/results_{exp}.csv'))
        # if exp is model_B or model_A, add a column 'trainfrac' with all the values 1
        if exp == 'model_B' or exp == 'model_A':
            df_exp['trainfrac'] = 1
        dfs.append(df_exp)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['circ'] >= circ_threshold]
    for trainfrac in df['trainfrac'].unique():
        df_sub = df[(df['trainfrac'] == trainfrac)]

        # Plot the first scatter plot
        sc1 = axes[0].scatter(df_sub['trainfrac'], df_sub['grad_sym'],
                            edgecolor='k', linewidth=0.2, s=40)
        axes[0].set_xlabel("Training Set Fraction")
        axes[0].set_ylabel("Gradient Symmetry")
        axes[0].set_xlim(-0.05, 1.05)
        axes[0].set_ylim(-0.05, 1.05)

        # Plot the first scatter plot
        sc2 = axes[1].scatter(df_sub['trainfrac'], df_sub['dist_irr'], 
                            edgecolor='k', linewidth=0.2, s=40)
        axes[1].set_xlabel("Training Set Fraction")
        axes[1].set_ylabel("Distance Irrelevance")
        axes[1].set_xlim(-0.05, 1.05)
        axes[1].set_ylim(-0.05, 1.05)

        # Plot the first scatter plot
        sc3 = axes[2].scatter(df_sub['trainfrac'], df_sub['circ'], 
                            edgecolor='k', linewidth=0.2, s=40)
        axes[2].set_xlabel("Training Set Fraction")
        axes[2].set_ylabel("Circularity")
        axes[2].set_xlim(-0.05, 1.05)
        axes[2].set_ylim(-0.05, 1.05)

    save_path = os.path.join(result_dir, f"grad_dist_frac_A.png" if experiment_name == 'model_A_frac' else f"grad_dist_frac_B.png")
    plt.savefig(save_path)
    plt.close()
