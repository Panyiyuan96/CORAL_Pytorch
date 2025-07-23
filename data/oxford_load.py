import os
import pickle
import numpy as np
import random
from PIL import Image
import torchvision.transforms as transforms
import torch

base_dir = "/Data/pointnetbase"
db_dir = "/Data/Oxford_DB"


def ele_input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def rgb_input_transform():
    return transforms.Compose([
        #transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

def RGB_dataloader(dict_value, queries, pos_num, neg_num):
    valid_data = 1

    if (len(dict_value["positives"]) < pos_num):
        valid_data = 0
        query_ = 0
        pos_ = 0
        neg_ = 0
        return valid_data, query_, pos_, neg_

    query = dict_value["query"]
    query_data = query["data"]
    query_name = query["name"]
    query_path = os.path.join(base_dir, query_data, "train_rgb", query_name)
    query_img = Image.open(query_path)
    query_ = rgb_input_transform()(query_img)

    positives = []
    for i in range(pos_num):
        pos = queries[dict_value["positives"][i]]["query"]
        pos_data = pos["data"]
        pos_name = pos["name"]
        pos_path = os.path.join(base_dir, pos_data, "train_rgb", pos_name)
        #print(pos_path)
        if os.path.isfile(pos_path):
            posImg = Image.open(pos_path)
            posTensor = rgb_input_transform()(posImg)

        positives.append(posTensor)
    pos_ = torch.stack(positives, 0)
  
    negatives = []
    for i in range(neg_num):
        neg = queries[dict_value["negatives"][i]]["query"]
        neg_data = neg["data"]
        neg_name = neg["name"]
        neg_path = os.path.join(base_dir, neg_data, "train_rgb", neg_name)

        if os.path.isfile(neg_path):
            negImg = Image.open(neg_path)
            negTensor = rgb_input_transform()(negImg)
        negatives.append(negTensor)
    neg_ = torch.stack(negatives, 0)

    return valid_data, query_, pos_, neg_

def ELE_dataloader(dict_value, queries, pos_num, neg_num):
    valid_data = 1

    if (len(dict_value["positives"]) < pos_num):
        valid_data = 0
        query_ = 0
        pos_ = 0
        neg_ = 0
        return valid_data, query_, pos_, neg_
    

    query = dict_value["query"]
    query_data = query["data"]
    query_name = query["name"]
    query_path = os.path.join(base_dir, query_data, "ele_filter", query_name)
    query_img = Image.open(query_path)
    query_ = ele_input_transform()(query_img)

    
    positives = []
    for i in range(pos_num):
        pos = queries[dict_value["positives"][i]]["query"]
        pos_data = pos["data"]
        pos_name = pos["name"]
        pos_path = os.path.join(base_dir, pos_data, "ele_filter", pos_name)
        #print(pos_path)
        if os.path.isfile(pos_path):
            posImg = Image.open(pos_path)
            posTensor = ele_input_transform()(posImg)
        positives.append(posTensor)
    pos_ = torch.stack(positives, 0)

    negatives = []
    for i in range(neg_num):
        neg = queries[dict_value["negatives"][i]]["query"]
        neg_data = neg["data"]
        neg_name = neg["name"]
        neg_path = os.path.join(base_dir, neg_data, "ele_filter", neg_name)

        if os.path.isfile(neg_path):
            negImg = Image.open(neg_path)
            negTensor = ele_input_transform()(negImg)
        negatives.append(negTensor)
    neg_ = torch.stack(negatives, 0)

    return valid_data, query_, pos_, neg_

def Multi_dataloader(dict_value, queries, pos_num, neg_num):
    valid_data = 1

    if (len(dict_value["positives"]) < pos_num):
        valid_data = 0
        rgb_query = 0
        rgb_pos = 0
        rgb_neg = 0
        ele_query = 0
        ele_pos = 0
        ele_neg = 0
        return valid_data, rgb_query, rgb_pos, rgb_neg, ele_query, ele_pos, ele_neg


    query = dict_value["query"]
    query_data = query["data"]
    query_name = query["name"]
    query_pathrgb = os.path.join(base_dir, query_data, "train_rgb", query_name)
    query_pathele = os.path.join(base_dir, query_data, "ele_filter", query_name)

    rgb_query = Image.open(query_pathrgb)
    ele_query = Image.open(query_pathele)

    rgb_query = rgb_input_transform()(rgb_query)
    ele_query = ele_input_transform()(ele_query)


    rgb_positives = []
    ele_positives = []

    for i in range(pos_num):
        pos = queries[dict_value["positives"][i]]["query"]
        pos_data = pos["data"]
        pos_name = pos["name"]
        pos_rgbpath = os.path.join(base_dir, pos_data, "train_rgb", pos_name)
        if os.path.isfile(pos_rgbpath):
            posImg = Image.open(pos_rgbpath)
            posTensor = rgb_input_transform()(posImg)
            rgb_positives.append(posTensor)

        pos_elepath = os.path.join(base_dir, pos_data, "ele_filter", pos_name)
        if os.path.isfile(pos_elepath):
            posImg = Image.open(pos_elepath)
            posTensor = ele_input_transform()(posImg)
            ele_positives.append(posTensor)

    rgb_pos = torch.stack(rgb_positives, 0)
    ele_pos = torch.stack(ele_positives, 0)



    rgb_negatives = []
    ele_negatives = []

    for i in range(neg_num):
        neg = queries[dict_value["negatives"][i]]["query"]
        neg_data = neg["data"]
        neg_name = neg["name"]
        neg_rgbpath = os.path.join(base_dir, neg_data, "train_rgb", neg_name)

        if os.path.isfile(neg_rgbpath):
            negImg = Image.open(neg_rgbpath)
            negTensor = rgb_input_transform()(negImg)
            rgb_negatives.append(negTensor)

        neg_elepath = os.path.join(base_dir, neg_data, "ele_filter", neg_name)
        if os.path.isfile(neg_elepath):
            negImg = Image.open(neg_elepath)
            negTensor = ele_input_transform()(negImg)
            ele_negatives.append(negTensor)

    rgb_neg = torch.stack(rgb_negatives, 0)
    ele_neg = torch.stack(ele_negatives, 0)


    return valid_data, rgb_query, rgb_pos, rgb_neg, ele_query, ele_pos, ele_neg



def DB_dataloader(dict_value):
    query = dict_value["query"]
    query_data = query["data"]
    query_name = query["name"]

    px = query_name.split('_')[1]
    py = query_name.split('_')[2]

    pose = torch.tensor([float(px), float(py)])

    query_pathrgb = os.path.join(db_dir, query_data, "train_rgb", query_name)
    query_pathele = os.path.join(db_dir, query_data, "ele_filter", query_name)

    rgb_query = Image.open(query_pathrgb)
    ele_query = Image.open(query_pathele)

    rgb_query = rgb_input_transform()(rgb_query)
    ele_query = ele_input_transform()(ele_query)

    return ele_query, rgb_query, pose

