from PIL import Image
import pickle
import random

import os, shutil
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
count = 0

x_width = 150
y_width = 150
p1 = [5735712.768124,620084.402381]
p2 = [5735611.299219,620540.270327]
p3 = [5735237.358209,620543.094379]
p4 = [5734749.303802,619932.693364]
p = [p1,p2,p3,p4]


def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    # print(northing)
    for point in points:
        if(point[0]-x_width < northing and northing < point[0]+x_width and point[1]-y_width < easting and easting < point[1]+y_width):
            in_test_set = True
            break
    return in_test_set

def filter_eleimg(Img):
    Img_narry = np.array(Img, dtype=np.uint8)
    Img_solid = Img_narry.copy()
    for i in range(1, 79):
        for j in range(1, 79):
            if Img_narry[i, j] > 4:
                continue
            else:
                Neighbor = Img_solid[i - 1:i + 2, j - 1:j + 2]
                Tol_height = np.sum(Neighbor)
                Num = np.sum(Neighbor > 4)
                if Num == 0:
                    continue
                else:
                    Img_narry[i, j] = Tol_height / Num

    Image_filter = Image.fromarray(Img_narry)
    return Image_filter


def filter_ele(base_path, folder, img_name):  # path为原始路径，disdir是移动的目标目录


    imgpath_ele = os.path.join(base_path, folder, "ele", img_name)

    filter_folder = os.path.join(base_path, folder, "ele_filter")

    if not os.path.exists(filter_folder):
        os.makedirs(filter_folder)

    if os.path.isfile(imgpath_ele):
        outpath = os.path.join(filter_folder, img_name)
        print(imgpath_ele, outpath)
        Img = Image.open(imgpath_ele)
        Img_filter = filter_eleimg(Img)
        Img_filter.save(outpath)

def create_resizergb(base_path, folder, img_name):  # path为原始路径，disdir是移动的目标目录


    imgpath_ele = os.path.join(base_path, folder, "img", img_name)

    filter_folder = os.path.join(base_path, folder, "train_rgb")

    if not os.path.exists(filter_folder):
        os.makedirs(filter_folder)

    if os.path.isfile(imgpath_ele):
        outpath = os.path.join(filter_folder, img_name)
        print(imgpath_ele, outpath)
        Img = Image.open(imgpath_ele)
        Img_resize = Img.resize((112,112))
        #Img_filter = filter_eleimg(Img)
        Img_resize.save(outpath)

def create_cat(sort_dir):
    data_list = os.listdir(sort_dir)

    for img_name in data_list:
        indice = img_name.split("__")[0]
        file_name = os.path.join(sort_dir, indice)
        if not os.path.exists(file_name):
            os.makedirs(file_name)

        img_path = os.path.join(sort_dir, img_name)
        img_movepath = os.path.join(file_name, img_name)
        shutil.move(img_path, img_movepath)

def is_invaild(Img):
    Img_narry = np.array(Img, dtype=np.uint8)

    if Img_narry[:30, :].any() == 0 and Img_narry[:55, :20].any() ==0 and Img_narry[:55, 60:].any() ==0:
        return True
    elif Img_narry[50, :].any() == 0 and Img_narry[25:, :20].any() ==0 and Img_narry[25:, 60:].any() ==0:
        return True
    else:
        return False

def construct_query_dict(df_centroids, filename):
    tree = KDTree(df_centroids[['X','Y']])
    ind_nn = tree.query_radius(df_centroids[['X','Y']],r=10)
    ind_r = tree.query_radius(df_centroids[['X','Y']], r=50)
    queries = {}
    print(len(ind_nn))
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]
        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
        random.shuffle(positives)
        negatives = np.setdiff1d(df_centroids.index.values.tolist(),ind_r[i]).tolist()
        random.shuffle(negatives)
        #positives = positives[:2]
        negatives = negatives[:200]
        queries[i] = {"query": query,
                      "positives": positives, "negatives": negatives}
        print(queries[i])

    with open(filename, 'wb') as handle:
         pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

if __name__ == '__main__':




    # create_cat(out_rgb_dir_n)

    # print(data_list)
    base_path = '/media/pyy/1T/bag/oxford/pointnetbase'
    data_list = os.listdir(base_path)
    all_folders = sorted(os.listdir(base_path))

    df_train = pd.DataFrame(columns=['data', 'name', 'X', 'Y'])

    folders = []
    for index in range(len(data_list)):
        folders.append(all_folders[index])

    print(len(folders))
    for folder in folders:
        img_folder_path = os.path.join(base_path, folder, "img")
        img_list =  os.listdir(img_folder_path)
        for img_name in img_list:
            p_x = float(img_name.split('_')[1])
            p_y = float(img_name.split('_')[2])
            if not (check_in_test_set(float(p_x)+5735176, float(p_y)+620398, p, x_width, y_width)):
                create_resizergb(base_path, folder, img_name)
                #filter_ele(base_path, folder, img_name)
                # Img_name = os.path.join(img_folder_path, img_name)
                # img = Image.open(Img_name)
                #if not (is_invaild(img)):
                # series = pd.Series({"data": folder, "name": img_name, "X": p_x, "Y": p_y})
                # df_train = df_train.append(series, ignore_index=True)

    #construct_query_dict(df_train,"oxford_training2.pickle")


