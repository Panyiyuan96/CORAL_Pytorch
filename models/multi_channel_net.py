import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

#input (3,224,224)
class ImageStream(nn.Module):
    def __init__(self, ResidualBlock):
        super(ImageStream, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.inchannel = 64

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #Residual block
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)

        #Top-Down
        self.tdlayer1 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)
        self.tdlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.tdlayer3 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.tdlayer4 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

        #Smooth
        self.smooth = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=1)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)


    def upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=[H,W], mode='bilinear') + y


    def forward(self, x):
        C1 = self.conv1(x)
        #C1 = self.maxpool(C1)
        R1 = self.layer1(C1)
        R2 = self.layer2(R1)
        R3 = self.layer3(R2)
        R4 = self.layer4(R3) # R4(512, 14, 14)

        #print(R4.shape)
        M4 = self.tdlayer4(R4)
        M3 = self.upsample_add(M4, self.tdlayer3(R3))
        M2 = self.upsample_add(M3, self.tdlayer2(R2))
        M1 = self.upsample_add(M2, self.tdlayer1(R1))

        M1 = self.smooth(M1)
        return M1
        # M1 (512, 112, 112)

class EleStream(nn.Module): #input (1, 80, 80)
    def __init__(self, ResidualBlock, trans_M):
        super(EleStream, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.inchannel = 64

        self.Length = 80

        #con3d

        self.convf2b = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        #Residual block
        self.layer1 = self.make_layer(ResidualBlock, 64,  1, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 192, 3, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, 3, stride=2)

        #Top-Down
        self.tdlayer1 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.tdlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.tdlayer3 = nn.Conv2d(192, 256, kernel_size=1, stride=1, padding=0)
        self.tdlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        #Project-Matrix
        self.Distance_M = torch.rand(2, 6400)
        for i in range(0, self.Length):
            for j in range(0, self.Length):
                self.Distance_M[0, i * self.Length + j] = (79.5 - i) * 0.5
                self.Distance_M[1, i * self.Length + j] = (39.5 - j) * 0.5
        self.Distance_M = self.Distance_M.cuda()
        #Transform layer

        self.trans_M = torch.from_numpy(trans_M).cuda()

        self.layer_num = 1
        #Multi perception
        multi1_input = 512 * self.layer_num
        multi1_num = 64
        self.multi1 = nn.Sequential(
            nn.Linear(multi1_input, multi1_num),
            nn.ReLU(),
            nn.Linear(multi1_num, multi1_num),
            nn.ReLU(),
            nn.Linear(multi1_num, multi1_num),
            nn.ReLU(),
            nn.Linear(multi1_num, multi1_num)
        )

        multi2_input = 512 * self.layer_num
        multi2_num = 128
        self.multi2 = nn.Sequential(
            nn.Linear(multi2_input, multi2_num),
            nn.ReLU(),
            nn.Linear(multi2_num, multi2_num),
            nn.ReLU(),
            nn.Linear(multi2_num, multi2_num),
            nn.ReLU(),
            nn.Linear(multi2_num, multi2_num)
        )

        multi3_input = 512 * self.layer_num
        multi3_num = 192
        self.multi3 = nn.Sequential(
            nn.Linear(multi3_input, multi3_num),
            nn.ReLU(),
            nn.Linear(multi3_num, multi3_num),
            nn.ReLU(),
            nn.Linear(multi3_num, multi3_num),
            nn.ReLU(),
            nn.Linear(multi3_num, multi3_num)
        )

        multi4_input = 512 * self.layer_num
        multi4_num = 256
        self.multi4 = nn.Sequential(
            nn.Linear(multi4_input, multi4_num),
            nn.ReLU(),
            nn.Linear(multi4_num, multi4_num),
            nn.ReLU(),
            nn.Linear(multi4_num, multi4_num),
            nn.ReLU(),
            nn.Linear(multi4_num, multi4_num)
        )

        #Smooth
        self.smooth = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)


    def transform_Matrix(self, multi_f, bv_img):
        layer_num = self.layer_num
        N, D, _, _ = bv_img.shape
        grid_num = self.Length * self.Length
        P_elep = torch.zeros([N, 4, grid_num * layer_num]).cuda()


        #Construct projection points based on the BEV img
        for n_layer in range(layer_num):
            P_elep[:, 0, n_layer * grid_num : (n_layer+1) * grid_num] = self.Distance_M[0, :].unsqueeze(0)
            P_elep[:, 1, n_layer * grid_num : (n_layer+1) * grid_num] = -self.Distance_M[1, :].unsqueeze(0)
            P_elep[:, 2, n_layer * grid_num : (n_layer+1) * grid_num] = -(bv_img.view(N, 1, -1).squeeze(1) * 255 * 0.02 - 2 - n_layer * 0.4)  #Converting grayscale values from an RGB image to real-world height
            P_elep[:, 3, n_layer * grid_num : (n_layer+1) * grid_num] = 1


        unvalid = torch.ones_like(P_elep[:, 2, :]) * -1000
        P_elep[:, 2, :] = torch.where(P_elep[:, 2, :] > 1.5, unvalid, P_elep[:, 2, :])
        B_Points = P_elep.permute(1, 0, 2).reshape(4, -1)
        B_pixels = torch.mm(self.trans_M, B_Points)
        B_pixels[0:2, :] = B_pixels[0:2, :] / B_pixels[2, :]
        B_pixels = B_pixels.view(3, -1, grid_num * layer_num)
        #
        zero = torch.zeros_like(B_pixels[0, :, :]).cuda()
        B_pixels[0, :, :] = torch.where(B_pixels[0, :, :] < 0, zero, B_pixels[0, :, :])
        B_pixels[0, :, :] = torch.where(B_pixels[0, :, :] > 1280, zero, B_pixels[0, :, :])
        B_pixels[1, :, :] = torch.where(B_pixels[0, :, :] == 0, zero, B_pixels[1, :, :]).int()
        B_pixels[1, :, :] = torch.where(B_pixels[1, :, :] < 0, zero, B_pixels[1, :, :])
        B_pixels[1, :, :] = torch.where(B_pixels[1, :, :] > 960, zero, B_pixels[1, :, :])
        B_pixels[0, :, :] = torch.where(B_pixels[1, :, :] == 0, zero, B_pixels[0, :, :]).int()
        #
        B_pixels[0, :, :] = B_pixels[0, :, :] / (1280 / 2) - 1
        B_pixels[1, :, :] = B_pixels[1, :, :] / (960 / 2) - 1
        B_pixels = B_pixels[:2, :, :].permute(1, 2, 0).view(N, 80, 80, -1)

        projected_img = F.grid_sample(multi_f, B_pixels, mode='bilinear', padding_mode='zeros')

        return projected_img



    def upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=[H,W], mode='bilinear') + y


    def forward(self, bv_img, multi_layer):
        C1 = self.conv1(bv_img)


        # B, _, _, _ = C1.shape
        # print(C1.shape)
        RGB_BVF = self.transform_Matrix(multi_layer, bv_img)
     

        #

        M1 = self.convf2b(RGB_BVF)+C1
  
        #
        #
        # RGB_F1 = F.interpolate(RGB_BVF, size=[40, 40], mode="bilinear")
        # # RGB_F2 = F.interpolate(RGB_BVF, size=[20, 20], mode="bilinear")
        # # RGB_F3 = F.interpolate(RGB_BVF, size=[10, 10], mode="bilinear")
        # # RGB_F4 = F.interpolate(RGB_BVF, size=[5, 5], mode="bilinear")
        #
        # RGB_F1 = RGB_F1.permute(0, 2, 3, 1).reshape(-1, 512 * self.layer_num)
        #
        #
        # R1 = self.layer1(C1).permute(0, 2, 3, 1).reshape(-1, 64)  # 1600*64
        # R1 = R1.view(B, 40, 40, -1).permute(0, 3, 1, 2)
        #
        R1 = self.layer1(M1)
        R2 = self.layer2(R1)
        R3 = self.layer3(R2)
        R4 = self.layer4(R3)

        M4 = self.tdlayer4(R4)
        M3 = self.upsample_add(M4, self.tdlayer3(R3))
        M2 = self.upsample_add(M3, self.tdlayer2(R2))
        # M1 = upsample_add(M2, self.tflayer4(R1))

        M2 = self.smooth(M2)
        return M2

class MultiNet(nn.Module):
    def __init__(self, fn_model, bn_model, net_vlad):
        super(MultiNet, self).__init__()
        self.fn_model = fn_model
        self.bn_model = bn_model
        self.net_vlad = net_vlad

    def forward(self, fv_img, bv_img):
        fv_features = self.fn_model(fv_img)

        #B,256,40,40
        multi_features = self.bn_model(bv_img, fv_features)
        # embedded_x = self.net_vlad(multi_features)
        embedded = self.net_vlad(multi_features)
        return embedded

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""
    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        weight_v = (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1).data
        self.conv.weight = nn.Parameter(weight_v)
        self.conv.bias = nn.Parameter(- self.alpha * self.centroids.norm(dim=1).data)



    def forward(self, x):
        N = x.size(0)
        C = x.size(1)

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
