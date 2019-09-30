import numpy as np
import torch
from os import listdir
from os.path import isfile, join
import csv
from PIL import Image

import sys
sys.path.append('functions')
from body_measurements import vertex2measurements
from render import mesh2Image, scale_mesh, transpose_mesh
from scale_mask import scale_fixed_height


def obj2Vertices(file):
    vertices = np.empty((0, 3), float)
    faces = np.empty((0, 3), int)
    for line in f:
        if line[:2] == 'v ':
            index1 = line.find(" ") + 1
            index2 = line.find(" ", index1 + 1)
            index3 = line.find(" ", index2 + 1)
            vertex = [[float(line[index1:index2]), float(
                line[index2:index3]), float(line[index3:-1])]]
            vertices = np.concatenate((vertices, vertex), axis=0)
        
        elif line[0] == 'f':
            string = line.replace("//", "/")
            i = string.find(" ") + 1
            face = []
            for item in range(string.count(" ")):
                if string.find(" ", i) == -1:
                    j = string.find('/',i)
                    face.append(int(string[i:j]) - 1)   # .obj vertices start from 1 but in opendr we start from 0 ,so -1
                    if string[i:j]!=string[j+1:-1]:
                        print('not matched!')
                        print(string[i:j])
                        print(string[j+1:-1])
                        input()
                    break
                j = string.find("/", i)
                face.append(int(string[i:j]) - 1)
                if string[i:j]!=string[j+1:string.find(' ', i)]:
                    print('not mached!')
                    print(string[i:j])
                    print(string[j+1,string.find(' ', i)])
                i = string.find(" ", i) + 1
            face = [face]
            faces = np.concatenate((faces,face), axis=0)
    return vertices, faces



path = '/home/yifu/workspace/data/MPI-FAUST/obj/male'
obj_files = [f for f in listdir(path) if isfile(join(path, f))]
csv_name = join(path, 'registration_measurements_gt.csv')
# m = load_model('../../models/basicModel_m_lbs_10_207_0_v1.0.0.pkl') 


with open(csv_name, mode='w') as csv_file:
    i = 0
    for name in obj_files:
        if name[-4:] != '.obj':
            print('not obj')
            continue
        print(name)
        i = i + 1
        fileName = join(path, name)
        f = open(fileName)
        vertices, faces = obj2Vertices(f)
        vertices_n = vertices.shape[0]
        print('vertices:',vertices_n)
        #m.r = vertices
        X, Y, Z = [vertices[:, 0], vertices[:, 1], vertices[:, 2]]
        X = torch.from_numpy(np.expand_dims(X, axis=0))
        Y = torch.from_numpy(np.expand_dims(Y, axis=0))
        Z = torch.from_numpy(np.expand_dims(Z, axis=0))

        height, waist, chest, neck, arm = vertex2measurements(X, Y, Z)
        csv_writer = csv.writer(
            csv_file, delimiter=',', lineterminator='\n')
        csv_writer.writerow([i, 'Name', name])
        csv_writer.writerow([i, 'Height', float(height)])
        csv_writer.writerow([i, 'Waist', float(waist)])
        csv_writer.writerow([i, 'Chest', float(chest)])
        csv_writer.writerow([i, 'Neck', float(neck)])
        csv_writer.writerow([i, 'Arm', float(arm)])
        csv_writer.writerow(['\n'])

        batch = 1
        height = 400
        width = 400
        img_height = 264
        img_width = 192 
        mesh = vertices
        name = name[:-4]

        #face = m.f
        faces = faces.astype(np.uint32)

        name = name + '_0'
        angle = np.pi
        axis = np.array([0, 0, 1])
        mesh = transpose_mesh(mesh, angle, axis)
        angle = np.pi
        axis = np.array([0, 1, 0])
        mesh = transpose_mesh(mesh, angle, axis)

        mesh_0 = scale_mesh(mesh, height, width, vertices_num = vertices_n)
        mesh2Image(mesh_0, faces, batch, path, name, height, width, vertices_num = vertices_n)
        image = Image.open(join(path,name+'.png'))
        output = scale_fixed_height(image,img_height,img_width)
        output_name = join(path, name+'.png')
        output.save(output_name)
        
        name = name[:-2] + '_1'
        angle = -np.pi/2
        axis = np.array([0, 1, 0])
        mesh = transpose_mesh(mesh, angle, axis)
        mesh_1 = scale_mesh(mesh, height, width,vertices_num = vertices_n)
        mesh2Image(mesh_1, faces, batch, path, name, height, width,vertices_num = vertices_n)
        image = Image.open(join(path,name+'.png'))
        output = scale_fixed_height(image,img_height,img_width)
        output_name = join(path, name+'.png')
        output.save(output_name)

csv_file.close()
