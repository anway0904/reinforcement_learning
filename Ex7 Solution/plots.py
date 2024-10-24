import matplotlib.pyplot as plt
import numpy as np

def line_plot(data:np.ndarray, num_episodes:int, num_trials:int, title:str):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = range(num_episodes)
    y = np.mean(data, axis=0)
    ax.plot(y)
    std_error = np.std(data, axis = 0)/np.sqrt(num_trials)
    ax.fill_between(x, y + 1.96*std_error, y - 1.96*std_error, alpha = 0.3)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    plt.show()