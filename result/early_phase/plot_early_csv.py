import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# read the csv file
# name = 'grad_sym'
# name = 'dist_irr'
# filename = name + '_B_2.csv'
# df = pd.read_csv(filename)


# # Clean the data
# # drop all the columns with its name contains __MIN and __MAX
# df = df.loc[:, ~df.columns.str.contains('__MIN|__MAX')]
# # the data is sampled every 50 Step, but some of them are missing, 
# # so we can take the maximum value of each column for each 50 steps
# df = df.groupby(df['Step'] // 50).max()

# # Step column -49 for each numbers
# df['Step'] = df['Step'] - 49
# # save the cleaned data
# df.to_csv(filename2, index=False)


# filename = 'dist_irr_B_d_64.csv'
# df1 = pd.read_csv(filename)

# filename = 'grad_sym_B_d_64.csv'
# df2 = pd.read_csv(filename)

# dim = 'd_64'
dim = 'wd_0.5'

df1 = pd.read_csv(f'dist_irr_B_{dim}.csv')

df2 = pd.read_csv(f'grad_sym_B_{dim}.csv')
 
df3 = pd.read_csv(f'norm_B_{dim}.csv')

# Clean the data
df1 = df1.loc[:, ~df1.columns.str.contains('__MIN|__MAX')]
df1 = df1.groupby(df1['Step'] // 50).max()
df1['Step'] = df1['Step'] - 49

df2 = df2.loc[:, ~df2.columns.str.contains('__MIN|__MAX')]
df2 = df2.groupby(df2['Step'] // 50).max()
df2['Step'] = df2['Step'] - 49

df3 = df3.loc[:, ~df3.columns.str.contains('__MIN|__MAX')]
df3 = df3.groupby(df3['Step'] // 50).max()
df3['Step'] = df3['Step'] - 49

steps = df1.iloc[:, 0]

# Data columns
data1, data2, data3 = df1.iloc[:, 1:], df2.iloc[:, 1:], df3.iloc[:, 1:]

plt.figure(figsize=(6, 4), dpi=300)
# Compute mean and confidence intervals
mean1 = data1.mean(axis=1)
confidence_interval1 = stats.sem(data1, axis=1) * stats.t.ppf((1 + 0.95) / 2., data1.shape[1]-1)
# smooth the confidence interval1
confidence_interval1 = np.convolve(confidence_interval1, np.ones(3)/2, mode='same')
confidence_interval1 = np.clip(mean1 + confidence_interval1, 0, 1) - mean1
# plt.plot(steps, mean1, label='Avg. Distance Irrelevance')
# plt.fill_between(steps, mean1 - confidence_interval1, mean1 + confidence_interval1, color='b', alpha=0.2, label='95% Confidence Interval')

# Compute mean and confidence intervals
mean2 = data2.mean(axis=1)
confidence_interval2 = stats.sem(data2, axis=1) * stats.t.ppf((1 + 0.95) / 2., data2.shape[1]-1)
# smooth the confidence interval2
confidence_interval2 = np.convolve(confidence_interval2, np.ones(3)/2, mode='same')
confidence_interval2 = np.clip(mean2 + confidence_interval2, 0, 1) - mean2
# plt.plot(steps, mean2, label='Avg. Gradient Symmetry')
# plt.fill_between(steps, mean2 - confidence_interval2, mean2 + confidence_interval2, color='orange', alpha=0.2, label='95% Confidence Interval')

mean3 = data3.mean(axis=1)
confidence_interval3 = stats.sem(data3, axis=1) * stats.t.ppf((1 + 0.95) / 2., data3.shape[1]-1)
confidence_interval3 = np.convolve(confidence_interval3, np.ones(3)/2, mode='same')
confidence_interval3 = np.clip(mean3 + confidence_interval3, 0, 100) - mean3

# def forward(a):
#     a = np.deg2rad(a)
#     return np.rad2deg(np.log(np.abs(np.tan(a) + 1.0 / np.cos(a))))

# def inverse(a):
#     a = np.deg2rad(a)
#     return np.rad2deg(np.arctan(np.sinh(a)))

def forward(x):
    return x**(1/2)


def inverse(x):
    return x**2

plt.scatter(0, 1.05, color='white', s=10)
plt.scatter(0, 0, color='white', s=10)
plt.xlabel('Steps')
plt.xscale('function', functions=(forward, inverse))
plt.xticks([0, 1000, 2500, 5000, 10000, 15000, 20000], ['0', '1k', '2.5k', '5k', '10k', '15k', '20k'])
# plt.legend(loc='lower right')

# plt.twinx()
# plt.plot(steps, mean3, label='Avg. Norm')
# plt.fill_between(steps, mean3 - confidence_interval3, mean3 + confidence_interval3, color='g', alpha=0.2, label='95% Confidence Interval')
# plt.ylabel('Norm')
# plt.ylim(0, 100)
# plt.legend(loc='lower right')
# plt.show()

# Get the current axes before calling twinx()
ax1 = plt.gca()

# Plot the first set of data (if any)
ax1.plot(steps, mean1, label='Avg. Distance Irrelevance')
ax1.fill_between(steps, mean1 - confidence_interval1, mean1 + confidence_interval1, color='b', alpha=0.2, label='95% Confidence Interval')

ax1.plot(steps, mean2, label='Avg. Gradient Symmetry')
ax1.fill_between(steps, mean2 - confidence_interval2, mean2 + confidence_interval2, color='orange', alpha=0.2, label='95% Confidence Interval')


# Create a second y-axis
ax2 = plt.twinx()

# Plot the second set of data
line1, = ax2.plot(steps, mean3, label='Avg. Parameter Norm', color='tab:green')
ax2.fill_between(
    steps,
    mean3 - confidence_interval3,
    mean3 + confidence_interval3,
    color='tab:green',
    alpha=0.2,
    label='95% Confidence Interval'
)
ax2.set_ylabel('Norm')
ax2.set_ylim(20, 60)

# Collect legend handles and labels from both axes
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Combine them and create a single legend
handles = handles1 + handles2
labels = labels1 + labels2
ax2.legend(handles, labels, loc='lower right')

plt.savefig(f'early_stage_norm_{dim}.png')
# plt.show()