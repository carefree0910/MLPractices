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
    img = np.full([height, width, 3], 255, dtype=np.uint8)

    ############################################################
    #                  Write your code here!                   #
    ############################################################

    pass

    ############################################################
    #                           End                            #
    ############################################################

    if show:
        cv2.imshow("Neural Network", img)
        cv2.waitKey(1)
    return img
