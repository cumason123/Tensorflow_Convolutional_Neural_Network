import matplotlib.pyplot as plt
import numpy as np
import math, os, cv2
FUNCTIONS_DIRECTORY_NAME = "funcDirectory"

def exponential(data):
    """Exponential Function

    :params data: list/np.array of x values

    :returns x^2
    """
    data = np.array(data)
    return data**2


def cubic(data):
    """Cubic Function

    :params data: list/np.array of x values

    :returns x^3
    """
    data=np.array(data)
    return data**3


def create_directory(directory):
    """Creates the directory if given string directory
    does not exist

    :params directory: string literal indicating directory name with path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_image_shape(filewithpath):
    return list(cv2.imread(filewithpath).shape)


def categorical(num_classes, classification):
    return [0]*classification + [1] + [0]*(num_classes-(classification+1))


def save_img(filewithpath, x, y, dpi=20):
    """Save an image

    :params filewithpath: string indicating file name with
    its pathing extension

    :params x: array of x data to be plotted

    :params y: array of y data to be plotted
    """
    fig = plt.figure()
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.plot(x,y,"k", linewidth=5)
    plt.savefig(filewithpath,dpi=dpi)
    plt.clf()


def genFunctions(directory, func):
    create_directory(directory)
    for iteration, i in enumerate(np.linspace(0,4,18)):
        if not i == 0:
            if not func==np.tan:
                x = np.linspace(-i*math.pi, i*math.pi, 10000)
                y = func(x)

                save_img(os.path.join(directory, func.__name__+str(iteration)+".jpeg"), x, y)
            else:
                x = np.linspace(-i*math.pi, i*math.pi, i*50)
                y = func(x)
                save_img(os.path.join(directory, func.__name__+str(iteration)+".jpeg"), x, y)


def gendata():
    """Generates dataset inside of FUNCTIONS_DIRECTORY_NAME
    """
    # functions = [np.sin, np.tan, np.arcsinh, np.arctanh, np.log, np.arccosh, np.arcsin,
    #              np.arctan, cubic, exponential, np.sinh, np.cosh, np.tanh]
    #
    # create_directory(FUNCTIONS_DIRECTORY_NAME)
    #
    # raw_x_vals = np.linspace(-40,40,10000)
    # x=list(raw_x_vals)
    # function_independents={}
    #
    # for func in functions:
    #     function_independents[func.__name__] = func(x)
    #
    # function_names = [name for name in function_independents]
    #
    # for filename in function_names:
    #     save_img(os.path.join(FUNCTIONS_DIRECTORY_NAME, filename+".jpeg"), x, list(function_independents[filename]))

    functions = [np.sin, np.tan, np.arcsinh, np.arctanh, np.log, np.arccosh, np.arcsin,
                 np.arctan, cubic, exponential, np.sinh, np.cosh, np.tanh]

    create_directory(FUNCTIONS_DIRECTORY_NAME)
    functionDirectories = [os.path.join(FUNCTIONS_DIRECTORY_NAME, func.__name__) for func in functions]

    for directory, func in np.array([functionDirectories, functions]).T:
        genFunctions(directory, func)



if __name__ == "__main__":
    gendata()

