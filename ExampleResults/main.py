import numpy as np
import matplotlib.pyplot as plt
from reader import extract
from graphic.visualization import Visualization
from spaces.local_space import LocalSpace
from spaces.world_space import WorldSpace, Camera
from PIL import Image
from time import time
import os


def render_with_shaders(world_space, visual_model, img):
    buffer = np.array(np.ones((visual_model.height, visual_model.width)) * -np.inf)
    for obj in world_space.objs:
        model_view = world_space.model_matrix()
        if world_space.camera.type_camera == 0:
            camera_matrix = world_space.orthographic_matrix()
        else:
            camera_matrix = world_space.perspective_matrix()
        obj.data['old_vertexes'] = np.zeros((len(obj.data['vertexes']), 4))
        data_face = {}
        for i in range(len(obj.data['edges'])):
            print(i)
            data_face['new_vertexes'], data_face['new_normals'], data_face['vertexes_before_ndc'] \
                = world_space.vertex_shader(obj, model_view, camera_matrix, i)
            visual_model.fill_triangle(obj, data_face, i, buffer, img)


if __name__ == '__main__':
    size = (512, 512)
    path = '../models'
    b = os.path.join(path,'floor.obj')
    a = os.path.join(path,'face.obj')
    c = os.path.join(path,'BRBC.obj')
    d = os.path.join(path,'eye_model.obj')
    eye_tex = Image.open('../models/eye_tex.tga')
    storm_tex = Image.open('../models/BRBC_tex.PNG')
    african_tex = Image.open('../models/african_head_diffuse.tga')
    floor_tex = Image.open('../models/floor_diffuse.tga')
    yoda = 'yoda.obj'
    name_file = a
    img = np.zeros(shape=(size[0] + 1, size[1] + 1, 3)).astype(np.uint8)
    face = LocalSpace(a, size=200, position=[0, 270, 0])
    floor = LocalSpace(b, size=200, position=[0, 100, 0])
    eye = LocalSpace(d, size=200, position=[0, 300, 0])
    stormtrooper = LocalSpace(c, size=200, position=[0, 400, 0],rot_y=90)


    camera = Camera([300, 300, 300],stormtrooper.data['position'])
    cam_front = Camera([0, 300, 500], face.data['position'])
    cam_top = Camera([0, 700, 1], face.data['position'])
    cam_bot = Camera([0, -700, 1], face.data['position'])
    cam_right = Camera([700, 100, 0], face.data['position'])
    cam_left = Camera([-700, 100, 0], face.data['position'])
    cam_back = Camera([0, 100, -400], face.data['position'])
    cam_orth = Camera([-100, 270, 0], face.data['position'], type_camera=0, height_view=(200, -250))


    objs = [stormtrooper]
    colors = [storm_tex]
    cam = camera
    WS = WorldSpace(objs, cam)
    s = time()
    # WS.pipeline_for_obj()
    vis = Visualization(img, objs, colors, cam.camera_position, type_shadows=0, type_model=1)
    render_with_shaders(WS, vis, img)
    # vis.show()
    print(time() - s)
    plt.imshow(np.rot90(img), cmap="gray", interpolation="none")
    plt.show()
