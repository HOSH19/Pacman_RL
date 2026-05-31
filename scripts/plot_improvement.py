# import argparse
import numpy as np
import os
from matplotlib import pyplot as plt

def calc_stats(filepath):
    data = np.load(filepath)["results"]
    # sort the arrays and delete the first and last elements
    data = np.sort(data, axis=1)
    data = np.delete(data, -1, axis=1)
    data = np.delete(data, 0, axis=1)
    avg = round(np.mean(data), 2)
    std = round(np.std(data), 2)
    return avg, std

# parser = argparse.ArgumentParser()
# parser.add_argument("-f", "--filepath", required=True, help="Specify the file path to the agent.", type=str)
# parser.add_argument("-s", "--save", help="Specify whether to save the chart.", action="store_const", const=True)
# args = parser.parse_args()

filepaths = []
agent_dirs = os.listdir("agents/")
agent_dirs.sort()
for d in agent_dirs:
    if "dqn_v2" in d:
        path = "agents/" + d + "/evaluations.npz"
        filepaths.append(path)

means = []
stds = []
for path in filepaths:
    avg, std = calc_stats(path)
    means.append(avg)
    stds.append(std)

runs = []
for i in range(len(filepaths)):
    runs.append(i + 1)
plt.xlabel("Training Run")
plt.ylabel("Score")
plt.bar(runs, means)
plt.bar(runs, stds)
plt.legend(["Mean evaluation score", "Standard deviation"])
plt.title("Average Evaluation Score and Standard Deviation\nAdjusted for Outliers   Agent: dqn_v2")
plt.show()
# plt.savefig("charts/fig1")


