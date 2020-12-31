from geometric_functions import affine_transformation as at
from reader import extract
import numpy as np


class LocalSpace:
    def __init__(self, name_file, size, position, rot_x=0, rot_y=0, rot_z=0):
        """
        Класс который хранит всю информацию об объекте. А также он имеет методы для перемещения объекта, поворота
         и увелечения.
        :param name_file: путь до файла
        :param size: число или тройка чисел, которые говорят во сколько раз увеличить
        :param position: тройка чисел, в которую нужно переместить объект
        :param rot_x: количество градусов для поворота относитель оси X
        :param rot_y: количество градусов для поворота относитель оси Y
        :param rot_z: количество градусов для поворота относитель оси Z
        """
        # создания словаря для хранения все информации об объекте
        self.data = {'size': size if type(size) == list else [size, size, size], 'position': position,
                     'rot_x': rot_x*np.pi/180, 'rot_y': rot_y*np.pi/180, 'rot_z': rot_z*np.pi/180}
        # считывание данных из файла
        self.__read_data_from_file(name_file)
        # преобразование к проективным координатам
        self.data['vertexes'] = at.vertexes_to_projective(self.data['vertexes'])
        self.data['normals'] = at.vertexes_to_projective(self.data['normals'])

        self.origin_data = self.data.copy()

    def __read_data_from_file(self, name_file):
        """
        Данная функция считывает из файла информацию об объекте
        :param name_file: путь до файла
        """
        self.data['vertexes'] = extract.get_something(name_file, 'v')
        self.data['edges'] = extract.get_faces(name_file)[0]
        self.data['textures'] = extract.get_something(name_file, 'vt')
        self.data['edges_textures'] = extract.get_faces(name_file)[1]
        self.data['normals'] = extract.get_something(name_file, 'vn')
        self.data['edges_normals'] = extract.get_faces(name_file)[2]

    def rotate_y(self, alpha, with_normals=False):
        """
        Данная функция выполняет поворот на угол alpha относительно оси Y. Поворот осуществеляется с помощью матрицы
        аффиного преобразования
        :param alpha: угол поворота в радианах
        :param with_normals: True - значит подобную операцию нужно провести также для нормалей объекта
        """
        self.data['vertexes'] = at.rotate_y(alpha, self.data['vertexes'])
        if with_normals:
            self.data['normals'] = at.rotate_y(alpha, self.data['normals'], True)

    def rotate_x(self, alpha, with_normals=False):
        """
        Данная функция выполняет поворот на угол alpha относительно оси X. Поворот осуществеляется с помощью матрицы
        аффиного преобразования
        :param alpha: угол поворота в радианах
        :param with_normals: True - значит подобную операцию нужно провести также для нормалей объекта
        """
        self.data['vertexes'] = at.rotate_z(alpha, self.data['vertexes'])
        if with_normals:
            self.data['normals'] = at.rotate_z(alpha, self.data['normals'], True)

    def rotate_z(self, alpha, with_normals=False):
        """
        Данная функция выполняет поворот на угол alpha относительно оси Z. Поворот осуществеляется с помощью матрицы
        аффиного преобразования
        :param alpha: угол поворота в радианах
        :param with_normals: True - значит подобную операцию нужно провести также для нормалей объекта
        """
        self.data['vertexes'] = at.rotate_z(alpha, self.data['vertexes'])
        if with_normals:
            self.data['normals'] = at.rotate_z(alpha, self.data['normals'], True)

    def scale(self, size, with_normals=False):
        """
        Данная функция выполняет увелечение в переденное количество раз. Увелечение осуществеляется с помощью матрицы
        аффиного преобразования
        :param size: тройка чисел, которые говорят во сколько раз увеличить
        :param with_normals: True - значит подобную операцию нужно провести также для нормалей объекта
        """
        self.data['vertexes'] = at.scaling(self.data['vertexes'], size)
        if with_normals:
            self.data['normals'] = at.scaling(self.data['normals'], size, True)

    def transfer(self, new_coords, with_normals=False):
        """
        Данная функция выполняет параллельный перенос. Перенос осуществеляется с помощью матрицы
        аффиного преобразования.
        :param new_coords: координаты точки куда нужно перенести объект
        :param with_normals: True - значит подобную операцию нужно провести также для нормалей объекта
        """
        # вычисляеются коэфициенты смещение для того, чтобы середина объекта находилась в точке new_coords
        k = [
            new_coords[0] - self.data['vertexes'][:, 0].mean(),
            new_coords[1] - self.data['vertexes'][:, 1].mean(),
            new_coords[2] - self.data['vertexes'][:, 2].mean()
        ]
        self.data['vertexes'] = at.parallel_translation(self.data['vertexes'], k)
        if with_normals:
            self.data['normals'] = at.parallel_translation(self.data['normals'], k, True)
