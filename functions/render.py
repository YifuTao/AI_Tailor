import time
import pickle
import torch
import cv2
from opendr.camera import ProjectPoints
from opendr.lighting import LambertianPointLight
from opendr.renderer import ColoredRenderer
import cv2
from numpy import random
import numpy as np
import sys
sys.path.append("/home/yifu/workspace/smpl")


def mesh2Image(vertices, faces, batch, path, name, height, width, vertices_num = 6890):
    # Create OpenDR renderer
    rn = ColoredRenderer()

    rt_1 = np.zeros(3)

    rn.camera = ProjectPoints(
        v=vertices,  # vertices
        # v=m,
        rt=rt_1,
        # x, y, z translation of the camera, z>=0    0 0 2
        t=np.array([0, 0, 0]),
        # f=np.array([w,w])/2, # focus length? just scaling the picture
        # c=np.array([w,h])/2, #  just move the picture along top-left axis? not sure
        f=np.array([1, 1]),
        c=np.array([0, 0]),
        k=np.zeros(5))
    rn.frustum = {'near': 1, 'far': 15, 'width': width, 'height': height}
    rn.set(v=vertices, f=faces, bgcolor=np.zeros(3))

    # Construct point light source
    rn.vc = LambertianPointLight(
        f=faces,  # face
        v=vertices,
        # v=rn.v, #vertex?
        num_verts=len(vertices),
        light_pos=np.array([-1000, -1000, -2000]),  # point light position
        vc=np.ones_like(vertices)*.9,  # albedo per vertex
        light_color=np.array([1., 1., 1.]))   # Blue, Green, Red; light intensity

    # make the image binary(black and white); these are actually magic steps
    rn.change_col(np.ones((vertices_num, 3)))
    #mask = rn.r.copy()  # takes lots of time
    
    mask = rn.r *255
    import cv2
    if batch == 1:
        cv2.imwrite('%s/%s.png' % (path, name), mask)
    else:
        cv2.imwrite('%s/%s_%d.png' % (path, name, i), mask)
    
    '''
    mask = mask[:, :, 0].astype(np.uint8)
    hand = rn.r.copy()*255.
    image = np.expand_dims(mask, 2) * hand
    
    import cv2
    if batch == 1:
        cv2.imwrite('%s/%s.png' % (path, name), image)
    else:
        cv2.imwrite('%s/%s_%d.png' % (path, name, i), image)
    '''


def transpose_mesh(vertices, angle, axis):
    rot = angle * axis
    R = cv2.Rodrigues(rot)[0]
    vertices = np.transpose(np.matmul(R, np.transpose(vertices)))
    return vertices


def scale_mesh(vertices, height, width, vertices_num = 6890):

    umax = np.max(vertices[:, 0])
    umin = np.min(vertices[:, 0])
    vmax = np.max(vertices[:, 1])
    vmin = np.min(vertices[:, 1])

    c = 2 * np.max([umax - umin, vmax - vmin])
    ss = height/c
    vertices = np.array([[ss, ss, 1], ]*vertices_num)*vertices

    umax = np.max(vertices[:, 0])
    umin = np.min(vertices[:, 0])
    vmax = np.max(vertices[:, 1])
    vmin = np.min(vertices[:, 1])

    tumax = height-umax  # or width?
    tumin = umin
    tvmax = width-vmax
    tvmin = vmin

    tu = (tumax-tumin)/2
    tv = (tvmax-tvmin)/2

    vertices = vertices + np.array([[tu, tv, 0], ]*vertices_num)

    umax = np.max(vertices[:, 0])
    umin = np.min(vertices[:, 0])
    vmax = np.max(vertices[:, 1])
    vmin = np.min(vertices[:, 1])

    vertices[:, 2] = 10.0 + (vertices[:, 2]-np.mean(vertices[:, 2]))
    vertices[:, :2] = vertices[:, :2] * np.expand_dims(vertices[:, 2], 1)
    return vertices


# parameters:3rot69pose10shape
def rendering(m, parameters, batch, path, name, height, width, angle, axis, vertices_num = 6890):

    for i in range(0, batch):
        if batch == 1:
            m.pose[:] = parameters[:72]   # 72pose+10shape=82
            m.betas[:] = parameters[72:]
        else:
            m.pose[:] = parameters[i][:72]
            m.betas[:] = parameters[i][72:]

        # to orthographic projection: a quick fix trick
        vertices = np.copy(m.r)
        faces = np.copy(m.f)

        vertices = transpose_mesh(vertices, angle, axis)
        vertices = scale_mesh(vertices, height, width, vertices_num=vertices_num)
        mesh2Image(vertices, faces, batch, path, name, height, width, vertices_num=vertices_num)
        


def main():
    # check the model's prediction
    import pickle
    batch_size = 2
    dataset_size = 4
    path = 'dataset/check'

    for n in range(dataset_size):
        for mode in ['predict', 'actual']:
            # unpickle parameters
            pickle_path = '%s/%s_%d_0' % (path, mode, n)
            #pickle_path = '%s/set_%d_%s'% (path,n, mode)
            infile = open(pickle_path, 'rb')
            parameters = pickle.load(infile)   # 72pose+10shape=82


if __name__ == "__main__":
    main()
