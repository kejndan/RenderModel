import numpy as np
from geometric_functions import affine_transformation as at


class Camera:
    def __init__(self, camera_pos, observ_point, type_camera=1, depth_view=(-1, -300), width_view=(300, -200),
                 height_view=(200, 0)):
        """
        Данный класс используется для хранения информацию о положение камеры, какую проекцию она использует, точки
        наблюдения и её границ.
        :param camera_pos: координаты позиции камеры
        :param observ_point: точка куда смотрит камера
        :param type_camera: 1 - перспективная проекция; 0 - ортографическая проекция
        :param depth_view: границы обзора камеры в глубину
        :param width_view: границы обзора камеры в ширину
        :param height_view: границы обзора камеры в высоту
        """
        self.camera_position = np.array(camera_pos)
        self.observation_point = np.array(observ_point)
        self.type_camera = type_camera
        self.depth_view = depth_view
        self.width_view = width_view
        self.height_view = height_view

    def get_edges(self):
        """
        Данная функция используется для быстрого получения границ обзора камеры, в классе WorldSpace
        :return:
        """
        return self.width_view[0], self.width_view[1], self.height_view[0], self.height_view[1], self.depth_view[0],\
               self.depth_view[1]


class WorldSpace:
    def __init__(self, objs, camera, size=(512, 512)):
        """
        Данный класс используется хранения объектов в мировой системе координат.
        В этом классе содержутся функции для полного пайплайна от локальных координат к экранным.
        :param objs: набор объектов
        :param camera: объект камера
        :param size: размеры экрана вывода
        """
        self.objs = objs
        self.camera = camera
        self.observation_point = camera.observation_point  # точка куда смотрит камера
        self.control_vec = np.array([0, 1, 0])
        self.width, self.height = size[0], size[1]
        self.__init_objects()  # перенос объектов из локальных координат в глобальные

    def pipeline_for_obj(self):
        """
        Данная функция реализует пайплайн перехода от мироквых координат к экранным для всех объектов.
        """
        self.__init_camera()
        if self.camera.type_camera == 0:
            self.__orthographic_view()
        else:
            self.__perspective_view()
        self.__screen_space()

    def __init_objects(self):
        """
        Переход от локальных к мировым координатам
        """
        for obj in self.objs:  # цикл по всем объектам
            obj.scale(obj.data['size'], with_normals=True)  # увеличиваем объект
            obj.rotate_x(obj.data['rot_x'], with_normals=True)  # поворачиваем по оси X
            obj.rotate_y(obj.data['rot_y'], with_normals=True)  # поворачиваем по оси Y
            obj.rotate_z(obj.data['rot_z'], with_normals=True)  # поворачиваем по оси Z
            # делаем паралльные перенос
            obj.transfer([obj.data['position'][0], obj.data['position'][1], obj.data['position'][2]], with_normals=True)

    def model_matrix(self):
        """
        Данная функция создаем матрицу для перехода к lookAt системе наблюдения путем перехода к новому базису.
        """
        temp_z = self.camera.camera_position - self.observation_point  # вектор расстояния от камеры до объекта
        temp_z = temp_z / np.linalg.norm(temp_z) if np.linalg.norm(temp_z) != 0 else temp_z  # нормируем данный вектор
        temp_x = np.cross(np.array([0, 1, 0]), temp_z)  # вектор перпедиклярный temp_z и temp_y
        temp_x = temp_x / np.linalg.norm(temp_x) if np.linalg.norm(temp_x) != 0 else temp_x # нормируем данный вектор
        temp_y = np.cross(temp_z, temp_x)  # считаем пердендикулярный вектор temp_x и temp_z
        # заполнение матрицы параллельного переноса и матрицы смены базиса
        trans_matrix = np.eye(4, 4)
        t_matrix = np.eye(4, 4)
        trans_matrix[0, :3] = temp_x
        trans_matrix[1, :3] = temp_y
        trans_matrix[2, :3] = temp_z
        t_matrix[:3, 3] = np.array([-self.camera.camera_position[0], -self.camera.camera_position[1],
                                    -self.camera.camera_position[2]])

        model_view = np.dot(trans_matrix, t_matrix)  # произведение матриц для получения model_matrix
        return model_view
    
    def orthographic_matrix(self):
        """
        Данная функция заполняет матрицу ортографической проекции.
        """
        r, l, t, b, f, n = self.camera.get_edges()  # получение границ обзора
        # получения ширины и высоты для сохранения пропроций
        self.dis_x = r - l
        self.dis_y = t - b

        ortho_matrix = np.array([[2 / (r - l), 0, 0, -((r + l) / (r - l))],
                                 [0, 2 / (t - b), 0, -((t + b) / (t - b))],
                                 [0, 0, 2 / (f - n), -((f + n) / (f - n))],
                                 [0, 0, 0, 1]])
        return ortho_matrix
    
    def perspective_matrix(self):
        """
        Данная функция заполняет матрицу перспективной проекции.
        """
        r, l, t, b, f, n = self.camera.get_edges()  # получение границ обзора
        # получения ширины и высоты для сохранения пропроций
        self.dis_x = r - l
        self.dis_y = t - b
        perspect_matrix = np.array([[2 * n / (r - l), 0, (r + l) / (r - l), 0],
                                   [0, 2 * n / (t - b), (t + b) / (t - b), 0],
                                   [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                                   [0, 0, -1, 0]])
        return perspect_matrix

    def __init_camera(self):
        """
        Данная функция делает переход к lookAt системе наблюдения.
        """
        model_view = self.model_matrix()
        for obj in self.objs:  # цикл по всем объектам
            # умножаем каждый вектор на model_matrix
            obj.data['vertexes'] = np.dot(model_view, obj.data['vertexes'].T).T
            # умножаем каждую нормаль на обратную и транспонированную model_matrix
            obj.data['normals'] = np.dot(np.linalg.inv(model_view).T, obj.data['normals'].T).T

    def __orthographic_view(self):
        """
        Данная функция делаем переход к NDC координатам через орфтографическую проекцию.
        """
        ortho_matrix = self.orthographic_matrix()
        for obj in self.objs:  # цикл по всем объектам
            obj.data['old_vertexes'] = obj.data['vertexes'].copy()  # сохраняем вектора до NDC координат
            obj.data['vertexes'] = np.dot(ortho_matrix, obj.data['vertexes'].T).T  # переходим к NDC координатам

    def __perspective_view(self):
        """
         Данная функция делаем переход к NDC координатам через перспективную проекцию.
         """
        perspect_matrix = self.perspective_matrix()
        for obj in self.objs:  # цикл по всем объектам
            obj.data['old_vertexes'] = obj.data['vertexes'].copy()  # сохраняем вектора до NDC координат
            # переходим к NDC координатам
            obj.data['vertexes'] = np.dot(perspect_matrix, obj.data['vertexes'].T).T
            obj.data['vertexes'][:, 0] /= -obj.data['vertexes'][:, 3]
            obj.data['vertexes'][:, 1] /= -obj.data['vertexes'][:, 3]
            obj.data['vertexes'][:, 2] /= obj.data['vertexes'][:, 3]

    def __to_center(self, obj):
        """
        Данная функция центрирует изображения в экранных координатах с помощью параллельного переноса.
        :param obj: объект который надо центрировать
        :return:
        """
        k = [
            self.width/2 - obj.data['vertexes'][:,0].mean(),
            self.height/2 - obj.data['vertexes'][:,1].mean(),
            0
        ]
        obj.data['vertexes'] = at.parallel_translation(obj.data['vertexes'], k)

    def __screen_space(self):
        """
        Данная функция реализует переход от NDC координат к экранным.
        """
        for obj in self.objs:  # цикл по всем объектам
            # растяжение объекта по оси X и Y
            obj.data['vertexes'][:, 0] = self.width / 2 * obj.data['vertexes'][:, 0] + self.width / 2
            obj.data['vertexes'][:, 1] = self.height / 2 * obj.data['vertexes'][:, 1] + self.height / 2
            # возращение пропроций объекту
            if self.dis_x > self.dis_y:
                obj.data['vertexes'][:, 1] = obj.data['vertexes'][:, 1] * self.dis_y / self.dis_x
            else:
                obj.data['vertexes'][:, 0] = obj.data['vertexes'][:, 0] * self.dis_x / self.dis_y
            # центрирование объекта
            # temp = obj.data['vertexes'][:, 3].copy()
            # obj.data['vertexes'][:, 3] = 1
            # self.__to_center(obj)
            # obj.data['vertexes'][:, 3] = temp
            obj.data['vertexes'] = np.rint(obj.data['vertexes']).astype(int)

    def vertex_shader(self, obj, model_view, camera_matrix, index):
        """
        Вершинный шейдер (преобразовывает координаты одного полигона)
        :param obj: объект чей полигон преобразовывается
        :param model_view: матрица перехода к lookAt системе
        :param camera_matrix: матрица проекции камеры
        :param index: индекс в полигона/нормали
        temp_vertexes: преобразованные вершины
        temp_normals: преобразованные нормали
        vertexes_before_ndc: вершины до перехода к NDC
        """
        # переход к lookAt системе
        temp_vertexes = np.dot(model_view, obj.data['vertexes'][obj.data['edges'][index]].T).T
        temp_normals = np.dot(np.linalg.inv(model_view).T, obj.data['normals'][obj.data['edges_normals'][index]].T).T

        # переход к NDC координатам
        vertexes_before_ndc = temp_vertexes.copy()
        temp_vertexes = np.dot(camera_matrix, temp_vertexes.T).T
        if self.camera.type_camera == 1:
            temp_vertexes[:, 0] /= -temp_vertexes[:, 3]
            temp_vertexes[:, 1] /= -temp_vertexes[:, 3]
            temp_vertexes[:, 2] /=  temp_vertexes[:, 3]

        # переход к экранным координатам
        temp_vertexes[:, 0] = self.width / 2 * temp_vertexes[:, 0] + self.width / 2
        temp_vertexes[:, 1] = self.height / 2 * temp_vertexes[:, 1] + self.height / 2
        if self.dis_x > self.dis_y:
            temp_vertexes[:, 1] = temp_vertexes[:, 1] * self.dis_y / self.dis_x
        else:
            temp_vertexes[:, 0] = temp_vertexes[:, 0] * self.dis_x / self.dis_y
        return np.rint(temp_vertexes).astype(int), temp_normals, vertexes_before_ndc
