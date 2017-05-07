import cv2
import imageio
import numpy as np

# What you should finished


def get_colors(lines, all_pos):
    """
    :param lines   : [ w_i1, w_i2, ..., w_ij, ... ]
    :param all_pos : Whether all weights in lines are positive
    :return        : Colors 
    """
    pos_color = [0, 195, 255]  # Color used when corresponding weight > 0
    neg_color = [255, 195, 0]  # Color used when corresponding weight < 0
    # noinspection PyTypeChecker
    colors = np.full([len(lines), 3], pos_color, dtype=np.uint8)  # Initialize colors with pos_color
    # If all_pos, return colors directly
    if all_pos:
        return colors.tolist()

    ############################################################
    #                  Write your code here!                   #
    ############################################################

    # Otherwise, switch to neg_color if corresponding weight < 0
    pass

    ############################################################
    #                           End                            #
    ############################################################

    return colors.tolist()


def get_line_info(weight, max_thickness=4, threshold=0.2):
    """
    :param weight        : [ [ w_11, w_12, ..., w_1j, ... ],
                             [ w_21, w_22, ..., w_2j, ... ],
                             ...,
                             [ w_i1, w_i2, ..., w_ij, ... ],
                             ...                             ]
    :param max_thickness : Max thickness of each line
    :param threshold     : Threshold of hidden lines
    :return              : (colors, thicknesses, masks)
    """
    # Scale 'weight' into [-1, 1]
    w_min, w_max = np.min(weight), np.max(weight)
    if w_min >= 0:
        weight -= w_min
        all_pos = True
    else:
        all_pos = False
    weight /= max(w_max, -w_min)
    masks = np.abs(weight) >= threshold
    # Get colors
    colors = [get_colors(lines, all_pos) for lines in weight]

    ############################################################
    #                  Write your code here!                   #
    ############################################################

    # Get thicknesses. Notice that each element in 'weight' could represent a 'ratio' now
    thicknesses = None

    ############################################################
    #                           End                            #
    ############################################################

    return colors, thicknesses, masks


def get_graphs(activations, neuron_block_width):
    """
    :param activations        : Activations
    :param neuron_block_width : Width & height of each neuron block 
    :return                   : Neuron graphs
    """
    graphs = []

    ############################################################
    #                  Write your code here!                   #
    ############################################################

    pass

    ############################################################
    #                           End                            #
    ############################################################

    return graphs


def place_graph(graphs, half_block_width, img, i, j, x, y):
    """
    Rendering neuron block
    :param graphs           : Neuron graphs
    :param half_block_width : int(neuron_block_width / 2)
    :param img              : Canvas
    :param i                : i-th hidden layer  
    :param j                : j-th neuron in i-th hidden layer
    :param x                : (x, y) is the center of the neuron graph on the canvas
    :param y                : (x, y) is the center of the neuron graph on the canvas
    :return                 : None 
    """
    ############################################################
    #                  Write your code here!                   #
    ############################################################

    pass

    ############################################################
    #                           End                            #
    ############################################################


def draw_circle(img, radius, x, y):
    """
    Render circle
    :param img    : Canvas 
    :param radius : Radius of the circle
    :param x      : (x, y) is the center of the circle graph on the canvas
    :param y      : (x, y) is the center of the circle graph on the canvas
    """
    ############################################################
    #                  Write your code here!                   #
    ############################################################

    pass

    ############################################################
    #                           End                            #
    ############################################################


def put_text(img, i, layers, y):
    """
    Put text on canvas
    :param img    : Canvas 
    :param i      : i-th hidden layer, notice that layers[i].name is the name of i-th hidden layer
    :param layers : Layers
    :param y      : (?, y) is the center of the neuron graph of i-th hidden layer 
    """
    ############################################################
    #                  Write your code here!                   #
    ############################################################

    pass

    ############################################################
    #                           End                            #
    ############################################################


def draw_line(img, i, j, k, x, y, new_x, new_y, half_block_width, colors, thicknesses):
    """
    Render line
    :param img              : Canvas
    :param i                : i-th weight matrix
    :param j                : j-th start-point neuron
    :param k                : k-th end-point neuron
        therefore [i][j][k] could pick up the corresponding color and thickness 
    :param x                : (x, y) is the start-point of the line
    :param y                : (x, y) is the start-point of the line
    :param new_x            : (new_x, new_y) is the end-point of the line
    :param new_y            : (new_x, new_y) is the end-point of the line
    :param half_block_width : int(neuron_graph_width / 2)
    :param colors           : Colors
    :param thicknesses      : Thicknesses
    :return: 
    """
    ############################################################
    #                  Write your code here!                   #
    ############################################################

    pass

    ############################################################
    #                           End                            #
    ############################################################


def bgr2rgb(im):
    """
    Switch BGR input image into RGB image
    :param im : Input image. Type: np.ndarray
    :return   : RGB image of input image 
    """
    ############################################################
    #                  Write your code here!                   #
    ############################################################

    return None

    ############################################################
    #                           End                            #
    ############################################################


# Wrappers

def draw_detail_network(show, x_min, x_max, layers, weights, get_activations):
    """
    :param show            : Whether show the frame or not 
    :param x_min           : np.minimum(data)
    :param x_max           : np.maximum(data)
    :param layers          : Layers in the network
    :param weights         : Weights in the network
    :param get_activations : Function which could return activations
                     Usage : get_activations(input_x, predict=True)
    :return                : 2d-visualization of the network 
    """
    radius = 3
    width = 1200
    height = 800
    padding = 0.2
    plot_scale = 2
    neuron_block_width = 30
    img = np.ones((height, width, 3), np.uint8) * 255

    n_layers = len(layers)
    units = [layer.shape[0] for layer in layers] + [layers[-1].shape[1]]

    if neuron_block_width % 2 == 1:
        neuron_block_width += 1
    half_block_width = int(neuron_block_width * 0.5)
    xf = np.linspace(x_min * plot_scale, x_max * plot_scale, neuron_block_width)
    yf = np.linspace(x_min * plot_scale, x_max * plot_scale, neuron_block_width) * -1
    input_x, input_y = np.meshgrid(xf, yf)
    input_xs = np.c_[input_x.ravel(), input_y.ravel()]

    activations = [activation.T.reshape(units[i + 1], neuron_block_width, neuron_block_width)
                   for i, activation in enumerate(get_activations(input_xs, predict=True))]
    graphs = get_graphs(activations, neuron_block_width)

    axis0_padding = int(height / (n_layers + 2 * padding)) * padding + neuron_block_width
    axis0 = np.linspace(
        axis0_padding,
        height - axis0_padding,
        n_layers + 1, dtype=np.int)
    axis1_padding = neuron_block_width
    axis1 = [np.linspace(axis1_padding, width - axis1_padding, unit + 2, dtype=np.int)
             for unit in units]
    axis1 = [axis[1:-1] for axis in axis1]

    colors, thicknesses, masks = [], [], []
    for weight in weights:
        line_info = get_line_info(weight)
        colors.append(line_info[0])
        thicknesses.append(line_info[1])
        masks.append(line_info[2])

    for i, (y, xs) in enumerate(zip(axis0, axis1)):
        for j, x in enumerate(xs):
            if i == 0:
                draw_circle(img, radius, x, y)
            else:
                place_graph(graphs, half_block_width, img, i-1, j, x, y)
        if i > 0:
            put_text(img, i-1, layers, y)

    for i, y in enumerate(axis0):
        if i == len(axis0) - 1:
            break
        for j, x in enumerate(axis1[i]):
            new_y = axis0[i + 1]
            for k, new_x in enumerate(axis1[i + 1]):
                if masks[i][j][k]:
                    draw_line(img, i, j, k, x, y, new_x, new_y, half_block_width, colors, thicknesses)

    if show:
        cv2.imshow("Neural Network", img)
        cv2.waitKey(1)
    return img


def make_mp4(ims, name="", fps=20):
    print("Making mp4...")
    with imageio.get_writer("{}.mp4".format(name), mode='I', fps=fps) as writer:
        for im in ims:
            writer.append_data(bgr2rgb(im))
    print("Done")
