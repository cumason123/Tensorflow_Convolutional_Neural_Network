import matplotlib.pyplot as plt
import numpy as np
import math, os
FUNCTIONS_DIRECTORY_NAME = "funcDirectory"

def exponential(data):
    data = np.array(data)
    return data**2


def cubic(data):
    data=np.array(data)
    return data**3

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_img(filewithpath, x, y, dpi=20):
    fig = plt.figure()
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.plot(x,y,"k", linewidth=5)
    plt.savefig(filewithpath,dpi=dpi)

    plt.clf()

functions = [np.sin, np.tan, np.arcsinh, np.arctanh, np.log, np.arccosh, np.arcsin,
             np.arctan, cubic, exponential, np.sinh, np.cosh, np.tanh]

def gendata():
    create_directory(FUNCTIONS_DIRECTORY_NAME)
    raw_x_vals = np.linspace(-40,40,10000)
    x=list(raw_x_vals)
    function_independents={}

    for func in functions:
        function_independents[func.__name__] = func(x)

    function_names = [name for name in function_independents]

    for filename in function_names:
        save_img(os.path.join(FUNCTIONS_DIRECTORY_NAME, filename+".jpeg"), x, list(function_independents[filename]))


if __name__ == "__main__":
    gendata()