import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
import kitti_data as dataset
import torch.optim as optim
from models.single_net import ResidualBlock, NetVLAD, EmbedNet, EleImgStream, ImageStream
from models.triplet_loss import HardTripletLoss
from data.oxford_load import Multi_dataloader, DB_dataloader, ELE_dataloader, RGB_dataloader
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from triplet_loss import Evaluation_dis
import numpy as np
import faiss
import math
import pickle

Pos_num = 2
Neg_num = 18
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
pickle_file = "/home/yiyuan/Desktop/code/oxford_training_all.pickle"

test_pickle = "/home/yiyuan/Desktop/code/oxford_test.pickle"
database_pickle = "/home/yiyuan/Desktop/code/oxford_database.pickle"

B=1

def get_queries_dict(filename):
    # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
    with open(filename, 'rb') as handle:
        queries = pickle.load(handle)
        print("Queries Loaded.")
        return queries

if __name__ == '__main__':
    writer = SummaryWriter()
	

    device = torch.device("cuda")
    Bn_stream = EleImgStream(ResidualBlock)
    Fn_stream = ImageStream(ResidualBlock)
    Net_vlad = NetVLAD(num_clusters=32, dim=512, alpha=1.0)

    model = EmbedNet(Fn_stream, Net_vlad).cuda()
    Cal_dis = Evaluation_dis(squared=False).cuda()
	# model = torch.load('/home/pyy/PycharmProjects/Multi-fuse/best_model.pkl')
    criterion = HardTripletLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    Best_Acc = 0
    channel_size = 8192*2

    training_queries = get_queries_dict(pickle_file)
    database_queries = get_queries_dict(database_pickle)
    testing_queries = get_queries_dict(test_pickle)
    
    train_num = len(training_queries)
    db_num = len(database_queries)
    test_num = len(testing_queries)
    print(len(training_queries), len(database_queries), len(testing_queries))
	
    for epoch in range(200):
        running_loss = 0
        epoch_loss = 0
        model.train()
        for iteration in range(train_num):
            query = training_queries[iteration]

            valid_data, rgb_query, rgb_pos, rgb_neg = RGB_dataloader(query, training_queries, Pos_num,
                                                                                        Neg_num)
            if valid_data == 0:
                continue
        
            N_p, C, H, W = rgb_pos.shape
            N_n, _, _, _ = rgb_neg.shape

            rgb_query = torch.unsqueeze(rgb_query, 0)

            pos = rgb_pos.view(B*N_p, C, H, W)
            neg = rgb_neg.view(B*N_n, C, H, W)
            traindata = torch.cat([rgb_query, pos, neg], 0).cuda()
            # print(traindata.shape)
            pos_label = torch.ones(N_p + 1).int()
            neg_label = torch.arange(2, 2+N_n, 1).int()
            labels = torch.cat((pos_label, neg_label), dim=0).cuda()

            optimizer.zero_grad()
            global_feature = model(traindata)
            # channel_size = global_feature.size(1)
            # distance = Cal_dis(global_feature)
            # print(distance)

            triplet_loss = criterion(global_feature, labels)
            triplet_loss.backward()
            optimizer.step()

            running_loss += triplet_loss.item()
            epoch_loss += triplet_loss.item()

            if iteration % 1000 == 999:  # print every 2000 mini-batches
                print("==> Epoch[{}]({}):Epoch Loss: {:.5f}".format(epoch, iteration, running_loss / 1000), flush=True)
                running_loss = 0.0
        print("====> Epoch[{}]:Epoch Loss: {:.5f}".format(epoch, epoch_loss), flush=True)

        if epoch % 1 == 0:
            Toal_TP = 0
            Toal_FP = 0
            Toal_TN = 0
            Toal_FN = 0
            with torch.no_grad():
                model.eval()

                # channel_size = 8192
                Feat_M = np.empty((db_num, channel_size))
                Pose_M = np.empty((db_num, 2))

                for iteration in range(db_num):
                    data = database_queries[iteration]
                    ele_query, rgb_query, pose = DB_dataloader(data)
                    #rgb_query = torch.unsqueeze(rgb_query, 0)
                    rgb_query = torch.unsqueeze(rgb_query, 0)
                    rgb_traindata = rgb_query.cuda()
                    #rgb_traindata = rgb_query.cuda()
                    global_feature = model(rgb_traindata)
                    # print(global_feature.shape)
                    Feat_M[iteration, :] = global_feature.detach().cpu().numpy()
                    Pose_M[iteration, :] = pose.numpy()

                Feat_M = Feat_M.astype('float32')
                DB_index = faiss.IndexFlatL2(channel_size)  # 构建FlatL2索引
                DB_index.add(Feat_M)

                all_data = 0
                print("Database has constructed!")
                correct = np.zeros((5, 2))
                n_values = [1, 25]
                correct_at_n = np.zeros(10)
                s_judge = [9, 25, 100, 225, 625]
                distance_25 = np.zeros(25)
                # for iteration, (ele_query, pose, index) in enumerate(data_loader2):
                #     ele_traindata = ele_query.cuda()
                #     global_feature = model(ele_traindata)
                #     print(global_feature.shape)
                #     Feat_M[index.detach().numpy(), :] = global_feature.detach().cpu().numpy()
                #     Pose_M[index.detach().numpy(), :] = pose.numpy()
                Tot_size = 0

                for iteration in range(test_num):
                    test_data = testing_queries[iteration]
                    ele_query, rgb_query, pose = DB_dataloader(test_data)
                    # rgb_query = torch.unsqueeze(rgb_query, 0)
                    rgb_query = torch.unsqueeze(rgb_query, 0)

                    rgb_traindata = rgb_query.cuda()
                    # rgb_traindata = rgb_query.cuda()
                    global_feature = model(rgb_traindata)
                    Query_size = global_feature.size(0)
                    Query = global_feature.detach().cpu().numpy()
                    Query = Query.astype('float32')
                    Poses = pose.numpy()
                    dis, pred = DB_index.search(Query, max(n_values) + 1)

                    # print(dis, pred)

                    Tot_size += Query_size

                    for i in range(Query_size):
                        for s_i in range(max(n_values)):
                            indice = pred[i, s_i + 1]
                            distance_25[s_i] = pow(Poses[0] - Pose_M[indice, 0], 2) + pow(Poses[1] - Pose_M[indice, 1], 2)

                    for index, S_distance in enumerate(s_judge):
                        for i in range(max(n_values)):
                            if (distance_25[i] < S_distance):
                                if (i < 1):
                                    correct[index][:] += 1
                                else:
                                    correct[index][1] += 1
                                break;

                for index, S_distance in enumerate(s_judge):
                    print("{:.0f}m===> Recall@1: {:.4f}; Recall@25: {:.4f}".format(
                        math.sqrt(S_distance), correct[index][0] / float(Tot_size), correct[index][1] / float(Tot_size)),
                        flush=True)


    print('Finished Training')
    writer.close()

