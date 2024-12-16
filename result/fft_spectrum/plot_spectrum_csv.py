import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Read the csv file
model_type = 'A'
root_path = f'D:/Downloads/new/'

# model_type = 'A'
# root_path = f'D:/Downloads/new/model_{model_type}_embeddings/embed/'

# model_type = 'B'
# root_path = f'D:/Downloads/new/model_{model_type}_embeddings_d_256/embed/'

# sp = []
# columns = None
# for i in range(13):
#     filename = f'spectrum_{model_type}_{i+1}.csv'
#     try:
#         df1 = pd.read_csv(root_path+filename)
#         spectrum = df1.iloc[-1].values
#         columns = df1.columns
#         sp.append(spectrum)
#     except:
#         print('Error reading file:', filename)
#         continue

# # save sp into pd.dataframe, then save to csv
# sp = pd.DataFrame(sp, columns=columns)
# sp.to_csv(root_path+f'spectrum_{model_type}_all.csv', index=False, sep=',')

# read from csv
sp = pd.read_csv(root_path+f'spectrum_{model_type}.csv', sep=',').values
sp_mean = np.mean(sp, axis=0)

freq = list(range(1, 30))
# Plot the spectrum using sns histogram
plt.figure(figsize=(5, 4))
plt.bar(freq, sp_mean)
plt.xlabel(r'Frequency $k$')
plt.ylabel(r'Norm $\|\mathbf{F}_k\|^2$')
plt.ylim(0, 18)
plt.show()
plt.close()

# Plot a signal based on the freq and sp (the amplitude)
# plt.figure(figsize=(10, 4))
# for j in range(len(sp)):
#     signal = np.zeros(1000)
#     for i in freq:
#         signal += sp[j, i-1]*np.sin(2*np.pi*i*np.linspace(0, 0.1, 1000))
#     plt.plot(signal, label=f'Signal {j+1}')
# # Plot a reference signal with a different frequency k=59
# reference = 50*np.sin(2*np.pi*59*np.linspace(0, 0.1, 1000)) 
# plt.plot(reference, label='Reference Signal')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()
# plt.close()
