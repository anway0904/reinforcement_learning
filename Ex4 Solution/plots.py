import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def plot_value_function(value_function, title:str):
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    X = np.arange(1, 11, 1)
    Y = np.arange(12, 22, 1)
    X, Y = np.meshgrid(X, Y)
    Z = value_function[12:22,1:,0]
    # ax.plot_surface(X, Y, Z, alpha = 0.9, antialiased = False)
    ax.plot_surface(X, Y, Z, alpha = 0.9, antialiased = False, cmap=cm.summer)
    ax.set_zlim(-1, 1)
    ax.set_xlim(1, 10)
    ax.set_ylim(12, 21)
    ax.set_ylabel("Player sum")
    ax.set_xlabel("Dealer showing")
    ax.set_title("No Usable Ace")

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    Z = value_function[12:22,1:,1]
    # ax.plot_surface(X, Y, Z, alpha = 0.9, antialiased = False, color = [0.2, 0.6, 0.43])
    ax.plot_surface(X, Y, Z, alpha = 0.9, antialiased = False, cmap=cm.summer)
    ax.set_zlim(-1, 1)
    ax.set_xlim(1, 10)
    ax.set_ylim(12, 21)
    ax.set_ylabel("Player sum")
    ax.set_xlabel("Dealer showing")
    ax.set_title("Usable Ace")

    fig.suptitle(title)
    plt.show()