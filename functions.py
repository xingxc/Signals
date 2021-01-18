# %%
import numpy as np
import matplotlib.pyplot as plt


def multiPolynomial(A, B):
    return 0


def plot_imag(num):
    num = np.array(num)
    origin = [0, 0]
    for i in range(num.real.__len__()):
        plt.plot([origin[0], num[i].real], [origin[1], num[i].imag], 'ro-')

    magnitude = max([np.linalg.norm(item) for item in num])
    plt.xlim([-magnitude, magnitude])
    plt.ylim([-magnitude, magnitude])
    plt.xlabel('Real')
    plt.ylabel('Image')
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.show()


# %%
