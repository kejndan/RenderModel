import numpy as np




def draw_vertexes(img, data_vertex):
    x_data = data_vertex[:, 0]
    y_data = data_vertex[:, 1]
    img[x_data, y_data] = [255, 255, 255]  # drawing of picture
    return img


def draw_edges(img, data_vertex, data_edges):
    """
    This function draws the edges
    """
    i = 0
    for v1, v2, v3 in data_edges:  # get the numbers of string
        # # v1, v2, v3 = v1 - 1, v2 - 1, v3 - 1  # change the numbering
        # print(v1,v2,v3)
        img = draw_line(img, data_vertex, v1, v2)
        img = draw_line(img, data_vertex, v1, v3)
        img = draw_line(img, data_vertex, v2, v3)
        i += 1
        # print(i)
    return img