import numpy as np


def centroid(a, b, c):
    """
    This function calculation the centroid of triangle with coords a,b,c
    """
    x = (a[0] + b[0] + c[0])/3
    y = (a[1] + b[1] + c[1])/3
    return [int(round(x)), int(round(y))]


def get_centroid_matrix(vertexes, data_edges):
    """
    This function calculation the centroid for each triangle in data_edges
    """
    centroid_matrix = np.empty(((len(data_edges)), 2))
    count = 0
    for v1, v2, v3 in data_edges:  # get the numbers of string
        v1, v2, v3 = v1 - 1, v2 - 1, v3 - 1  # change the numbering
        centroid_matrix[count] = centroid(vertexes[v1], vertexes[v2], vertexes[v3])
        count += 1
    return centroid_matrix


def get_square_matrix(data_vertexes, data_edges):
    """
    This function calculation the squares for each triangle in data_edges
    The squares calculation through the vector product
    """
    square_matrix = np.empty((len(data_edges)))
    count = 0
    for v1, v2, v3 in data_edges:  # get the numbers of string
        v1, v2, v3 = v1 - 1, v2 - 1, v3 - 1  # change the numbering
        a = [data_vertexes[v1][0], data_vertexes[v2][0], data_vertexes[v3][0]]
        b = [data_vertexes[v1][1], data_vertexes[v2][1], data_vertexes[v3][1]]
        square_matrix[count] = np.linalg.norm((np.cross(a, b)))//2
        count += 1
    return square_matrix


def center_mass(centroids, squares):
    """
    :param centroids: this centroids are each triangle in picture
    :param squares: this squares are each triangle in picture
    """
    s = np.sum(squares)
    x_centr = np.sum(centroids[:, 0]*squares)
    y_centr = np.sum(centroids[:, 1]*squares)
    return np.array([[x_centr/s, y_centr/s]])


def equation_plane(v1, v2, v3):
    p1 = np.array([v1[0], v1[1], v1[2]])
    p2 = np.array([v2[0], v2[1], v2[2]])
    p3 = np.array([v3[0], v3[1], v3[2]])
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1,v2)
    a, b, c = cp
    d = np.dot(cp, p3)
    return a, b, c, d

def get_barr_coords(x, y, a, b, c):
    ab = [b[0] - a[0], b[1] - a[1]]
    ac = [c[0] - a[0], c[1] - a[1]]
    pa = [a[0] - x, a[1] - y]
    temp = np.array([ab, ac, pa]).transpose()
    v_prod = np.cross(temp[0], temp[1])
    if v_prod[2] != 0:
        v_prod = v_prod / v_prod[2]
        v_prod[2] = 1 - v_prod[0] - v_prod[1]
        return np.array([v_prod[2], v_prod[0], v_prod[1]])
    else:
        return np.array([-1, -1, -1])

