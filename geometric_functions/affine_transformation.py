import numpy as np


def vertexes_to_projective(vertexes):
    return np.concatenate([vertexes.copy(), np.ones(vertexes.shape[0]).reshape(-1, 1)], axis=1)


def parallel_translation(coord_matrix, t, for_normal=False):
    shift_matrix = np.array([
        [1, 0, 0, t[0]],
        [0, 1, 0, t[1]],
        [0, 0, 1, t[2]],
        [0, 0, 0, 1]
    ])
    if for_normal:
        return np.dot(np.linalg.inv(shift_matrix).T, coord_matrix.T).T
    else:
        return np.dot(shift_matrix, coord_matrix.T).T


def rotate_z(alpha, coord_matrix, for_normal=False):
    turn_matrix = np.array([[np.cos(2*np.pi - alpha), -np.sin(2*np.pi - alpha), 0, 0],
                            [np.sin(2*np.pi - alpha), np.cos(2*np.pi - alpha), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]
                            ])

    if for_normal:
        return np.dot(np.linalg.inv(turn_matrix).T, coord_matrix.T).T
    else:
        return np.dot(turn_matrix, coord_matrix.T).T


def rotate_y(alpha, coord_matrix, for_normal=False):
    turn_matrix = np.array([[np.cos(2*np.pi - alpha), 0, np.sin(2*np.pi - alpha), 0],
                            [0, 1, 0, 0],
                            [-np.sin(2*np.pi - alpha), 0, np.cos(2*np.pi - alpha), 0],
                            [0, 0, 0, 1]
                            ])

    if for_normal:
        return np.dot(np.linalg.inv(turn_matrix).T, coord_matrix.T).T
    else:
        return np.dot(turn_matrix, coord_matrix.T).T


def rotate_x(alpha, coord_matrix, for_normal=False):
    turn_matrix = np.array([[1, 0, 0, 0],
                            [0, np.cos(2*np.pi - alpha), -np.sin(2*np.pi - alpha), 0],
                            [0, np.sin(2*np.pi - alpha), np.cos(2*np.pi - alpha), 0],
                            [0, 0, 0, 1]
                            ])

    if for_normal:
        return np.dot(np.linalg.inv(turn_matrix).T, coord_matrix.T).T
    else:
        return np.dot(turn_matrix, coord_matrix.T).T


def scaling(coord_matrix, size, for_normal=False):
    scal_matrix = np.array([
        [size[0], 0, 0, 0],
        [0, size[1], 0, 0],
        [0, 0, size[2], 0],
        [0, 0, 0, 1]
    ])
    if for_normal:
        return np.dot(np.linalg.inv(scal_matrix).T, coord_matrix.T).T
    else:

        return np.dot(scal_matrix, coord_matrix.T).T


if __name__ == '__main__':
    print(parallel_translation((5, 5), (0, 0)))
