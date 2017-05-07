import cv2
import numpy as np

from NN.Source.Basic.Layers import SubLayer


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
    colors[lines < 0] = neg_color

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
    thicknesses = np.array(
        [[int((max_thickness - 1) * abs(n)) + 1 for n in lines] for lines in weight]
    )

    ############################################################
    #                           End                            #
    ############################################################

    return colors, thicknesses, masks


def draw_detail_network(show, x_min, x_max, layers, weights, get_activations):
    """
    :param show            : Whether show the frame or not 
    :param x_min           : np.minimum(x)
    :param x_max           : np.maximum(x)
    :param layers          : Layers in the network
    :param weights         : Weights in the network
    :param get_activations : Function which could return activations
                     Usage : get_activations(input_x, predict=True)
    :return                : 2d-visualization of the network 
    """
    # Constants which might be useful
    radius = 6
    width = 1200
    height = 800
    padding = 0.2
    plot_scale = 2
    plot_precision = 0.03
    img = np.ones((height, width, 3), np.uint8) * 255

    ############################################################
    #                  Write your code here!                   #
    ############################################################

    n_layers = len(layers)
    units = [layer.shape[0] for layer in layers] + [layers[-1].shape[1]]
    whether_sub_layers = np.array([False] + [isinstance(layer, SubLayer) for layer in layers])
    n_sub_layers = np.sum(whether_sub_layers)  # type: int

    plot_num = int(1 / plot_precision)
    if plot_num % 2 == 1:
        plot_num += 1
    half_plot_num = int(plot_num * 0.5)
    xf = np.linspace(x_min * plot_scale, x_max * plot_scale, plot_num)
    yf = np.linspace(x_min * plot_scale, x_max * plot_scale, plot_num) * -1
    input_x, input_y = np.meshgrid(xf, yf)
    input_xs = np.c_[input_x.ravel(), input_y.ravel()]

    activations = [activation.T.reshape(units[i + 1], plot_num, plot_num)
                   for i, activation in enumerate(get_activations(input_xs, predict=True))]
    graphs = []
    for j, activation in enumerate(activations):
        graph_group = []
        for ac in activation:
            data = np.zeros((plot_num, plot_num, 3), np.uint8)
            mask = ac >= np.average(ac)
            data[mask], data[~mask] = [0, 165, 255], [255, 165, 0]
            graph_group.append(data)
        graphs.append(graph_group)

    axis0_padding = int(height / (n_layers + 2 * padding)) * padding + plot_num
    axis0_step = (height - 2 * axis0_padding) / (n_layers + 1)
    axis0 = np.linspace(
        axis0_padding,
        height + n_sub_layers * axis0_step - axis0_padding,
        n_layers + 1, dtype=np.int)
    axis0 -= int(axis0_step) * np.cumsum(whether_sub_layers)
    axis1_padding = plot_num
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
                cv2.circle(img, (x, y), radius, (20, 215, 20), int(radius / 2))
            else:
                graph = graphs[i - 1][j]
                img[y - half_plot_num:y + half_plot_num, x - half_plot_num:x + half_plot_num] = graph
        if i > 0:
            cv2.putText(img, layers[i - 1].name, (12, y - 36), cv2.LINE_AA, 0.6, (0, 0, 0), 1)

    for i, y in enumerate(axis0):
        if i == len(axis0) - 1:
            break
        for j, x in enumerate(axis1[i]):
            new_y = axis0[i + 1]
            whether_sub_layer = isinstance(layers[i], SubLayer)
            for k, new_x in enumerate(axis1[i + 1]):
                if whether_sub_layer and j != k:
                    continue
                if masks[i][j][k]:
                    cv2.line(img, (x, y + half_plot_num), (new_x, new_y - half_plot_num),
                             colors[i][j][k], thicknesses[i][j][k])

    ############################################################
    #                           End                            #
    ############################################################

    if show:
        cv2.imshow("Neural Network", img)
        cv2.waitKey(1)
    return img
