from random import randint
import numpy as np
from geometric_functions.geometry_calculations import equation_plane, get_barr_coords


class Visualization:
    def __init__(self, img, objs, color_maps=None, camera_p=None, light_p=None,z_buffer=True,
                 back_face_culling=True, type_model=2, texture=True, type_shadows=1, size=(513, 513)):
        """
        Данная функция растеризует объект.
        :param img: матрица отбражения
        :param objs: объекты отрисовки
        :param color_maps: текстуры для объектов
        :param camera_p: точка расположения камеры
        :param light_p: точка располжения света
        :param z_buffer: True - z-buffer включен, False - z-buffer выключен
        :param back_face_culling: True - back-face culling включен, False - back-face culling  выключен
        :param type_model: 0 - объекты не освещяются, 1 - объекты освещяются по Ламберту, 2 - объекты освещяются по Фонгу
        :param texture: True - текстуры включены, False - текстуры выключены
        :param type_shadows: 0 - затенение через flat shading, 1 - затенение по Фонгу
        :param size: размеры экрана вывода для буфера
        """
        self.img = img
        self.objs = objs
        if light_p is None:
            self.light_point = np.array([0, 1000, 1000])
        else:
            self.light_point = np.array(light_p)
        self.camera_point = camera_p

        self.width = size[0]
        self.height = size[1]

        self.color_maps = color_maps
        self.back_face_culling = back_face_culling
        self.type_model = type_model
        self.texture = texture
        self.type_shadows = type_shadows
        self.z_buffer = z_buffer

        self.buffer = np.array(np.ones((self.height, self.width)) * -np.inf)
        # коэффиценты для фонового освещения
        self.ambient_strength = .1  # сила фонового освещения
        self.ambient_k = 1  # свойство текстуры для фонового освещения

        # коэффиценты для дифузного освещения
        self.diffuse_strength = 1  # сила дифузного освещения
        self.diffuse_k = 1  # свойство текстуры для дифузного освещения

        # коэффиценты для зеркального освещения
        self.specular_strength = .3  # сила зеркального освещения
        self.specular_k = 1  # свойство текстуры для зеркального освещения

    def show(self):
        """
        Растеризация всех объектов
        """
        for obj in self.objs:  # цикл по всем объектам
            # убираем четвертую координату
            obj.data['original_vertexes'] = obj.data['vertexes'].copy()
            obj.data['vertexes'] = obj.data['vertexes'][:, :3]
            obj.data['old_vertexes'] = obj.data['old_vertexes'][:, :3]
            for i in range(len(obj.data['edges'])):  # цикл по всем полигонам
                # индексы вершин полигона
                a = obj.data['edges'][i, 0]
                b = obj.data['edges'][i, 1]
                c = obj.data['edges'][i, 2]

                # считаем нормаль к данному полигону и нормируем её
                norm_face = np.cross(obj.data['old_vertexes'][c] - obj.data['old_vertexes'][a],
                                     obj.data['old_vertexes'][c] - obj.data['old_vertexes'][b])
                norm_face = norm_face / np.linalg.norm(norm_face)

                print(i, [obj.data['vertexes'][a], obj.data['vertexes'][b], obj.data['vertexes'][c]])
                # если back-face culling включен, то проверяем нужно ли оторбражать эту грань
                # если он не включен или грань нужно оторбражать, то идём дальше
                if self.back_face_culling:
                    if np.dot(norm_face, np.array([0, 0, -1])) >= 0:
                        continue
                # создаем матрицу трех вершин и находим её минимумы и максимумы по каждой оси
                vector = self.__get_vector(obj, a, b, c)
                x_max, y_max = vector.max(axis=0)
                x_min, y_min = vector.min(axis=0)
                # проходимся по полученному прямоугольнику
                for x in range(x_min, x_max + 1):
                    for y in range(y_min, y_max + 1):
                        if 0 <= x <= 512 and 0 <= y <= 512: # если x и y лежат внутри экрана оторбражения
                            # считаем барицентрические координаты для заданной точкиэкранной проекции, после чего
                            # получаем их для мировых координат
                            barr_coords = get_barr_coords(x, y, obj.data['vertexes'][a],
                                                          obj.data['vertexes'][b],
                                                          obj.data['vertexes'][c])

                            s = np.array([barr_coords[0] / obj.data['original_vertexes'][a][3],
                                          barr_coords[1] / obj.data['original_vertexes'][b][3],
                                          barr_coords[2] / obj.data['original_vertexes'][c][3]])
                            p_z = 1 / s.sum()
                            barr_clip = s * p_z
                            # если точка лежит внутри треуголника
                            if np.all(barr_coords >= 0):
                                # путем интерполяции получаем z координату для данной точки
                                z = np.dot(barr_clip, [obj.data['old_vertexes'][a][2],
                                                         obj.data['old_vertexes'][b][2],
                                                         obj.data['old_vertexes'][c][2]])

                                # если z-buffer включен, то проверяем нужно ли рисовать данный пиксель
                                # если он не включен или пиксель нужно отображать то идём дальше
                                if self.z_buffer:
                                    if self.buffer[x, y] < z:
                                        self.buffer[x, y] = z
                                    else:
                                        continue
                                # получаем цвет данного пикселя
                                color = self.__get_color(obj, i, barr_clip)

                                if self.type_shadows == 0:  # если затенение через flat shading
                                    # получаем середину данного полигона
                                    point = np.array(([obj.data['old_vertexes'][a],
                                                       obj.data['old_vertexes'][b],
                                                        obj.data['old_vertexes'][c]])).sum(axis=0) / 3
                                    # получаем уровень освещения данного полигона
                                    result_light = self.__get_light_lvl(norm_face, point)
                                else: # если затенение по Фонгу
                                    # считаем нормаль к данному пикселю путем интерполяции
                                    n1 = obj.data['normals'][obj.data['edges_normals'][i, 0]]
                                    n2 = obj.data['normals'][obj.data['edges_normals'][i, 1]]
                                    n3 = obj.data['normals'][obj.data['edges_normals'][i, 2]]
                                    n_x = np.dot(barr_clip, [n1[0], n2[0], n3[0]])
                                    n_y = np.dot(barr_clip, [n1[1], n2[1], n3[1]])
                                    n_z = np.dot(barr_clip, [n1[2], n2[2], n3[2]])
                                    norm_pixel = np.array([n_x, n_y, n_z]) / np.linalg.norm(np.array([n_x, n_y, n_z]))
                                    # получаем уровень освещения данного пикселя
                                    result_light = self.__get_light_lvl(norm_pixel, [x, y, z])
                                # умножаем уровень света на цвет после чего записываем в матрицу отображения
                                self.img[x, y] = np.dot(result_light, color[:3])

    def fragment_shader(self, obj, data, index, buffer, x, y):
        """
        Фрагментный шейдер (закрашивает нужный пиксель)
        :param obj: объект визуализации
        :param data: информация о вершинах, нормалях полигона
        :param index: номер полигона
        :param buffer: z-buffer для данного окна визуализации
        :param x: координата x
        :param y: координата y
        :return: если пиксель не нужно оторбражать, то [-1, -1, -1], иначе цвет пикселя
        """
        # считаем барицентрические координаты для заданной точки экранной проекции, после чего
        # получаем их для мировых координат
        barr_coords = get_barr_coords(x, y, data['new_vertexes'][0],
                                      data['new_vertexes'][1],
                                      data['new_vertexes'][2])
        s = np.array([barr_coords[0] / data['original_vertexes'][0],
                      barr_coords[1] / data['original_vertexes'][1],
                      barr_coords[2] / data['original_vertexes'][2]])
        p_z = 1 / s.sum()
        barr_clip = s * p_z
        # если точка лежит внутри треуголника
        if np.all(barr_coords >= 0):
            # путем интерполяции получаем z координату для данной точки
            z = np.dot(barr_clip, [data['vertexes_before_ndc'][0][2],
                                  data['vertexes_before_ndc'][1][2],
                                  data['vertexes_before_ndc'][2][2]])
            # если z-buffer включен, то проверяем нужно ли рисовать данный пиксель
            # если он не включен или пиксель нужно отображать то идём дальше
            if self.z_buffer:
                if buffer[x, y] < z:
                    if buffer[x,y] != -np.inf:
                        print(buffer[x,y], z)
                    buffer[x, y] = z
                else:
                    return np.array([-1, -1, -1])
            # получаем цвет данного пикселя
            color = self.__get_color(obj, index, barr_clip)

            if self.type_shadows == 0: # если затенение через flat shading
                # получаем середину данного полигона
                point = np.array([data['vertexes_before_ndc'][0],
                                  data['vertexes_before_ndc'][1],
                                  data['vertexes_before_ndc'][2]]).sum(axis=0) / 3
                # получаем уровень освещения данного полигона
                result_light = self.__get_light_lvl(data['norm_face'], point)
            else:  # если затенение по Фонгу
                # считаем нормаль к данному пикселю путем интерполяции
                n1 = data['new_normals'][0]
                n2 = data['new_normals'][1]
                n3 = data['new_normals'][2]
                n_x = np.dot(barr_clip, [n1[0], n2[0], n3[0]])
                n_y = np.dot(barr_clip, [n1[1], n2[1], n3[1]])
                n_z = np.dot(barr_clip, [n1[2], n2[2], n3[2]])
                norm_pixel = np.array([n_x, n_y, n_z]) / np.linalg.norm(np.array([n_x, n_y, n_z]))
                # получаем уровень освещения данного пикселя
                result_light = self.__get_light_lvl(norm_pixel, [x, y, z])
            # умножаем уровень света на цвет после чего возвращеем его
            return np.dot(result_light, color[:3])
        else:
            return np.array([-1, -1, -1])
        
    def fill_triangle(self, obj, data, index, buffer, img):
        """
        Функция отрисовка полигона, которая вызывает фрагментный шейдер
        :param obj: объект визуализации
        :param data: информация о вершинах, нормалях полигона
        :param index: номер полигона
        :param buffer: z-buffer для данного окна визуализации
        :param img: матрица отображения
        """
        # убираем четвертую координату
        data['original_vertexes'] = data['new_vertexes'][:, 3].copy()
        data['new_vertexes'] = data['new_vertexes'][:, :3].copy()
        data['vertexes_before_ndc'] = data['vertexes_before_ndc'][:, :3]

        # считаем нормаль к данному полигону и нормируем её
        norm_face = np.cross(data['vertexes_before_ndc'][2] - data['vertexes_before_ndc'][0],
                             data['vertexes_before_ndc'][2] - data['vertexes_before_ndc'][1])
        data['norm_face'] = norm_face / np.linalg.norm(norm_face)

        # если back-face culling включен, то проверяем нужно ли оторбражать эту грань
        # если он не включен или грань нужно оторбражать, то идём дальше
        if self.back_face_culling:
            if np.dot(norm_face, np.array([0, 0, -1])) >= 0:
                return
        x_max, y_max = np.max(data['new_vertexes'][:, :2], axis=0)
        x_min, y_min = np.min(data['new_vertexes'][:, :2], axis=0)
        # проходимся по полученному прямоугольнику
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                if 0 <= x <= 512 and 0 <= y <= 512: # если x и y лежат внутри экрана оторбражения
                    # получаем цвет данного пикселя
                    color = self.fragment_shader(obj, data, index, buffer, x, y)
                    if np.all(color >= 0): # если функция вернула цвет пикселя, то рисуем его
                        img[x, y] = np.rint(color)

    def __get_color(self, obj,index,barr_coords):
        """
        Данная функция извелекает из текстуры цвет конкретной точки путем интерполяции
        :param obj: объект визуализации
        :param index: номер вершины
        :param barr_coords: барицентрические координаты для этой точки
        :return: цвет
        """
        if self.texture: # если оторбражение текстуры включенно
            # получаем координты для получения цвета у вершин полигона
            u1, v1 = obj.data['textures'][obj.data['edges_textures'][index, 0], :2]
            u2, v2 = obj.data['textures'][obj.data['edges_textures'][index, 1], :2]
            u3, v3 = obj.data['textures'][obj.data['edges_textures'][index, 2], :2]
            # получаем координаты цвета нужной нам координты
            u = np.dot(barr_coords, [u1, u2, u3])
            v = np.dot(barr_coords, [v1, v2, v3])
            # разрешения текстуры
            w = self.color_maps[self.objs.index(obj)].size[0]
            h = self.color_maps[self.objs.index(obj)].size[1]
            # получаем цвет данного пикселя
            color = self.color_maps[self.objs.index(obj)].getpixel((u * w - 1,
                                             h - 1 - v *
                                             h))
        else:  # если текстура выключена, то используем стандартный цвет
            color = np.array([127,127,127])
        return color

    def __get_light_lvl(self, normal, point):
        """
        Данная функция возвращает уровень освещенность для переданной точки
        :param normal: нормаль к это точки
        :param point: точка для которой считаем освещенность
        :return: уровень освещенности
        """
        if self.type_model == 0: # если освещение выключено
            result_light = 1
        elif self.type_model == 1: # если освещение по Ламберту
            result_light = self.__ambient() + self.__diffuse(normal, point)
        else:  # если освещение по Фонгу
            result_light = self.__ambient() + self.__diffuse(normal, point) \
                           + self.__specular(normal, point, 2)
        return result_light

    def __get_vector(self, obj, a, b, c):
        """
        Данная функция создает матрицу для трех вершин
        """
        vector = np.array([
            [obj.data['vertexes'][a][0], obj.data['vertexes'][a][1]],
            [obj.data['vertexes'][b][0], obj.data['vertexes'][b][1]],
            [obj.data['vertexes'][c][0], obj.data['vertexes'][c][1]]
        ]).astype(int)
        return vector

    # def __get_pixel_normal(self, obj,index, barr_coords):
    #     n1 = obj.data['normals'][obj.data['edges_normals'][index, 0]]
    #     n2 = obj.data['normals'][obj.data['edges_normals'][index, 1]]
    #     n3 = obj.data['normals'][obj.data['edges_normals'][index, 2]]
    #     n_x = np.dot(barr_coords, [n1[0], n2[0], n3[0]])
    #     n_y = np.dot(barr_coords, [n1[1], n2[1], n3[1]])
    #     n_z = np.dot(barr_coords, [n1[2], n2[2], n3[2]])
    #     pixel_normal = np.array([n_x, n_y, n_z]) / np.linalg.norm(np.array([n_x, n_y, n_z]))
    #     return pixel_normal

    def __ambient(self):
        """
        Данная функция считает фоновое освещение путем умножения силы данного освещения на свойство материала
        """
        return self.ambient_strength * self.ambient_k

    def __diffuse(self, normal, point):
        """
        Данная функция считает диффузное освещение
        :param normal: нормаль к точке
        :param point: точка переданная для освещещения
        :return неотрицательный косинус между нормалью и направлением луча света умноженного на силу данного типа света
        и его свойство материала
        """
        light_dir = self.light_point - point  # направления луча света
        return max(np.dot(normal, light_dir/np.linalg.norm(light_dir)), 0) *self.diffuse_strength*self.diffuse_k

    def __specular(self, normal, point, alpha):
        """
        Данная функция считает зеркальное освещение
        :param normal: нормаль к точке
        :param point: точка переданная для освещещения
        :param alpha: параметр площади блика
        :return неотрицательный косинус между направлением на наблюдателя и направлением отраженного луча умноженного на
        а силу данного типа света и его свойство материала
        """
        light_dir = self.light_point - point  # направления луча света
        reflect = -(2 * normal * np.dot(-light_dir, normal) + light_dir) # направление отраженного луча
        view_dir = self.camera_point - point  # направление на наблюдателя
        return self.specular_strength * self.specular_k * \
               max(np.dot(reflect/np.linalg.norm(reflect), view_dir/np.linalg.norm(view_dir)), 0) ** alpha


    def draw_edges(self):
        for v1, v2, v3 in self.objs[0].data['edges']:  # get the numbers of string
            self.__draw_line(v1, v2)
            self.__draw_line(v1, v3)
            self.__draw_line(v2, v3)
        return self.img



    def __draw_line(self, start, end):
        x0, y0 = self.objs[0].data['vertexes'][start][0], self.objs[0].data['vertexes'][start][1]
        x1, y1 = self.objs[0].data['vertexes'][end][0], self.objs[0].data['vertexes'][end][1]

        # get distance
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        # definition format of the line
        steep = False
        if dy > dx:  # to do the line more broad than long
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            dx, dy = dy, dx
            steep = True
        if x0 > x1:  # to do the line from left to right
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        err = 0
        d_err = dy  # value of the slope
        y = y0
        # definition signum y
        sgn_y = y1 - y0
        sgn_y = 1 if sgn_y > 0 else -1

        # to draw the line

        for x in range(x0, x1 + 1):
            if 0 <= x <= 512 and 0 <= y <= 512:
                # draw the point
                if steep:
                    self.img[y, x] = [255, 255, 255]
                else:
                    self.img[x, y] = [255, 255, 255]
                err += d_err  # increasing the error
                # if the error has exceeded the allowable norm
                if 2 * err >= dx:
                    y += sgn_y
                    err -= dx
        return self.img

