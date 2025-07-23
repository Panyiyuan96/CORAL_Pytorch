import os
import torch
import time
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.multi_channel_net import ResidualBlock, ImageStream, EleStream, NetVLAD, MultiNet
from models.triplet_loss import HardTripletLoss, Evaluation_dis
from data.oxford_load import Multi_dataloader, DB_dataloader
import kitti_data as dataset
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import numpy as np
import faiss
import math
import pickle

def get_queries_dict(filename):
    # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
    with open(filename, 'rb') as handle:
        queries = pickle.load(handle)
        print("Queries Loaded.")
        return queries

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

Pos_num = 2
Neg_num = 18
pickle_file = "/home/yiyuan/Code/oxford_training_all.pickle"
test_pickle = "/home/yiyuan/Code/oxford_test.pickle"
database_pickle = "/home/yiyuan/Code/oxford_database.pickle"

B=1

if __name__ == '__main__':
    writer = SummaryWriter('log')

    trans_M = np.array(([[0.00042768, -0.999967, -0.0081, -0.0119846],
                             [-0.00721, 0.008081, -0.999941, -0.05404],
                             [0.999974, 0.00048595, -0.007207, -0.292197],
                             [0, 0, 0, 1]]))
    batch_num = 1
    device = torch.device("cuda")
    Fn_stream = ImageStream(ResidualBlock)
    Bn_stream = EleStream(ResidualBlock, trans_M)
    Net_vlad = NetVLAD(num_clusters=32, dim=256, alpha=1.0)
    # FFT = FFT2()
    model = MultiNet(Fn_stream, Bn_stream, Net_vlad).cuda()

    # model = MultiNet(Fn_stream, Bn_stream, FFT).cuda()

    #model = torch.load('/home/yiyuan/Code/Multi-fuse/multi_model.pkl')
    criterion = HardTripletLoss().cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

  
    Best_Acc = 0
    channel_size = 8192
    print('Start Training')

    training_queries = get_queries_dict(pickle_file)
    database_queries = get_queries_dict(database_pickle)
    testing_queries = get_queries_dict(test_pickle)
    train_num = len(training_queries)
    db_num = len(database_queries)
    test_num = len(testing_queries)
    for epoch in range(100):
        running_loss = 0
        epoch_loss = 0
        model.train()
     
        for iteration in range(train_num):

        
            #torch.cuda.synchronize()
            #start_time = time.time()

            query = training_queries[iteration]
        
            valid_data, rgb_query, rgb_pos, rgb_neg, ele_query, ele_pos, ele_neg = Multi_dataloader(query, training_queries, Pos_num,
                                                                                        Neg_num)
            if valid_data == 0:
                continue
            #torch.cuda.synchronize()
            #data_time = time.time()
            N_p, C, H, W = rgb_pos.shape
            N_n, _, _, _ = rgb_neg.shape
        
            rgb_query = torch.unsqueeze(rgb_query, 0)
            rgb_pos = rgb_pos.view(B*N_p, C, H, W)
            rgb_neg = rgb_neg.view(B*N_n, C, H, W)
            rgb_traindata = torch.cat([rgb_query, rgb_pos, rgb_neg], 0).cuda()
            # print(rgb_query.shape, rgb_pos.shape, rgb_neg.shape, rgb_traindata.shape)
        
            _, C2, H2, W2 = ele_pos.shape
            ele_query = torch.unsqueeze(ele_query, 0)
            ele_pos = ele_pos.view(B * N_p, C2, H2, W2)
            ele_neg = ele_neg.view(B * N_n, C2, H2, W2)
            ele_traindata = torch.cat([ele_query, ele_pos, ele_neg], 0).cuda()
            # print(ele_query.shape, ele_pos.shape, ele_neg.shape, ele_traindata.shape)
            pos_label = torch.ones(N_p + 1).int()
            neg_label = torch.arange(2, 2 + N_n, 1).int()
            labels = torch.cat((pos_label, neg_label), dim=0).cuda()
        
            optimizer.zero_grad()
        
            global_feature = model(rgb_traindata, ele_traindata)
        
            triplet_loss = criterion(global_feature, labels)
            triplet_loss.backward()
            optimizer.step()

            #torch.cuda.synchronize()
            #optim_time = time.time()
            #print("dataload_process:{}, train_process:{}", data_time - start_time, optim_time - data_time)
        
            running_loss += triplet_loss.item()
            epoch_loss += triplet_loss.item()
            #print(iteration)
            if iteration % 1000 == 999:  # print every 2000 mini-batches
                print("==> Epoch[{}]({}):Epoch Loss: {:.5f}".format(epoch, iteration, running_loss / 1000), flush=True)
                running_loss = 0.0
        print("====> Epoch[{}]:Epoch Loss: {:.5f}".format(epoch, epoch_loss), flush=True)
        # writer.add_scalar("scalar/epoch_loss", epoch_loss, epoch)

        if epoch % 1 == 0:
            print("current model save!")
            torch.save(model, 'multi_model.pkl')
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
                    rgb_query = torch.unsqueeze(rgb_query, 0)
                    ele_query = torch.unsqueeze(ele_query, 0)
                    ele_traindata = ele_query.cuda()
                    rgb_traindata = rgb_query.cuda()
                    global_feature = model(rgb_traindata, ele_traindata)
                    # print(global_feature.shape)
                    Feat_M[iteration, :] = global_feature.detach().cpu().numpy()
                    Pose_M[iteration, :] = pose.numpy()

                Feat_M = Feat_M.astype('float32')
                DB_index = faiss.IndexFlatL2(channel_size) 
                DB_index.add(Feat_M)

                all_data = 0
                print("Database has constructed!")
                correct = np.zeros((5, 2))
                n_values = [1, 25]
                correct_at_n = np.zeros(10)
                s_judge = [9, 25, 100, 225, 625]
                distance_25 = np.zeros(25)
                Tot_size = 0

                for iteration in range(test_num):
                    test_data = testing_queries[iteration]
                    ele_query, rgb_query, pose = DB_dataloader(test_data)
                    rgb_query = torch.unsqueeze(rgb_query, 0)
                    ele_query = torch.unsqueeze(ele_query, 0)

                    ele_traindata = ele_query.cuda()
                    rgb_traindata = rgb_query.cuda()
                    global_feature = model(rgb_traindata, ele_traindata)
                    Query_size = global_feature.size(0)
                    Query = global_feature.detach().cpu().numpy()
                    Query = Query.astype('float32')
                    Poses = pose.numpy()
                    dis, pred = DB_index.search(Query, max(n_values) + 1)

                    #print(iteration)
                    #print(Poses[0], Poses[1])
                    #print("------")
                    Tot_size += Query_size

                    for i in range(Query_size):
                        for s_i in range(max(n_values)):
                            indice = pred[i, s_i + 1]
                            # print(indice)
                            # print(Poses[i, 0])
                            # print(Pose_M[indice, 1])
                            #if s_i == 0:
                                #print("s_i:") 
                                #print(Pose_M[indice, 0], Pose_M[indice, 1])
                            distance_25[s_i] = pow(Poses[0] - Pose_M[indice, 0], 2) + pow(Poses[1] - Pose_M[indice, 1], 2)
                    #print("------")
                    for index, S_distance in enumerate(s_judge):
                        for i in range(max(n_values)):
                            if (distance_25[i] < S_distance):
                                if (i < 1):
                                    correct[index][:] += 1
                                else:
                                    correct[index][1] += 1
                                break;
                print(Tot_size)
                for index, S_distance in enumerate(s_judge):
                    print("{:.0f}m===> Recall@1: {:.4f}; Recall@25: {:.4f}".format(
                        math.sqrt(S_distance), correct[index][0] / float(Tot_size), correct[index][1] / float(Tot_size)),
                        flush=True)

        print('Finished Training')
        writer.close()

