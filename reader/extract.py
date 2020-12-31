import numpy as np


def get_something(name_file, name_type):
    name_type += ' '
    f = open(name_file, 'r')
    file = f.read()
    part_file = file[file.find(name_type):]
    part_file = part_file.split('\n')
    vertexes = []
    for i in range(len(part_file)):
        if part_file[i].startswith(name_type):
            vertexes.append([float(s) for s in part_file[i].split()[1:]])
    return np.array(vertexes)


def get_faces(name_file):
    f = open(name_file, 'r')
    file = f.read()
    part_file = file[file.find('f '):]
    part_file = part_file.split('\n')
    faces = [[], [], []]
    for i in range(len(part_file)):
        if part_file[i].startswith('f '):
            line = part_file[i].split()[1:]
            faces[0].append([int(s.split('/')[0]) - 1 for s in line])
            faces[1].append([int(s.split('/')[1]) - 1 if s.split('/')[1] != '' else '' for s in line])
            faces[2].append([int(s.split('/')[2]) - 1 if s.split('/')[2] != '' else '' for s in line])
    return np.array(faces)
