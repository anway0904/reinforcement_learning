import matplotlib.pyplot as plt

def draw_plots(moving_average, retn, loss, lens, env_name):
    fig = plt.figure(figsize=(10,8))
    alph = 0.6
    avg_line_clr = 'b'


    ax = plt.subplot(3, 1, 1)
    ax.plot(retn, alpha = alph)
    ax.plot(moving_average(retn), color = avg_line_clr)
    ax.set_title(f"Returns ({env_name})")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")

    ax = plt.subplot(3, 1, 2)
    ax.plot(loss, alpha = alph)
    ax.plot(moving_average(loss), color = avg_line_clr)
    ax.set_title(f"Losses ({env_name})")
    ax.set_xlabel("steps")
    ax.set_ylabel("loss")

    ax = plt.subplot(3, 1, 3)
    ax.plot(lens, alpha = alph)
    ax.plot(moving_average(lens), color = avg_line_clr)
    ax.set_title(f"Lengths of Episode ({env_name})")
    ax.set_xlabel("episodes")
    ax.set_ylabel("length")

    fig.tight_layout(pad=3)
    plt.show()