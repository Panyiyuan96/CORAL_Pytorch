import torch
import torch.nn as nn
import torch.nn.functional as F
import time
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

class ImageStream(nn.Module):
    def __init__(self, ResidualBlock):
        super(ImageStream, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

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
        R1 = self.layer1(C1)
        R2 = self.layer2(R1)
        R3 = self.layer3(R2)
        R4 = self.layer4(R3)

        #print(R4.shape)
        M4 = self.tdlayer4(R4)
        M3 = self.upsample_add(M4, self.tdlayer3(R3))
        M2 = self.upsample_add(M3, self.tdlayer2(R2))
        M1 = self.upsample_add(M2, self.tdlayer1(R1))

        M1 = self.smooth(M1)

        return R4
        # out(n, 512, 128, 256)

class EleImgStream(nn.Module):
    def __init__(self, ResidualBlock):
        super(EleImgStream, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #Residual block
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 128, 4, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 192, 8, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, 8, stride=2)

        #Top-Down
        self.tdlayer1 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.tdlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.tdlayer3 = nn.Conv2d(192, 256, kernel_size=1, stride=1, padding=0)
        self.tdlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        #Smooth
        self.smooth = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)


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
        R1 = self.layer1(C1)
        R2 = self.layer2(R1)
        R3 = self.layer3(R2)
        R4 = self.layer4(R3)

        #print(R4.shape)
        M4 = self.tdlayer4(R4)
        M3 = self.upsample_add(M4, self.tdlayer3(R3))
        M2 = self.upsample_add(M3, self.tdlayer2(R2))
        M1 = self.upsample_add(M2, self.tdlayer1(R1))

        M1 = self.smooth(M1)

        return M1



class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        input_c = 256
        output_c = 256

        self.local_f = 0
        self.fft_f = 0
        self.mlp = nn.Sequential(
            nn.Linear(input_c, output_c),
            nn.ReLU(),
            nn.Linear(output_c, output_c),
            nn.ReLU(),
            nn.Linear(output_c, output_c),
        )

    def forward(self, x):
        x = self.base_model(x)
        # self.local_f = x
        #print(x.shape)
        out_put = self.net_vlad(x)
        # self.fft_f = embedded_x
        # pool_fea = self.pool(embedded_x).squeeze(-1).squeeze(-1)
        # # output = output.unsqueeze(-1)
        # out_put = self.mlp(pool_fea)
        # out_put = F.normalize(out_put, p=2, dim=1)

        return out_put

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
        embedded_x = self.net_vlad(multi_features)
        return embedded_x




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

def fftshift2d(x):
    for dim in range(2, len(x.size())):
        n_shift = x.size(dim)//2
        if x.size(dim) % 2 != 0:
            n_shift = n_shift + 1  # for odd-sized images
        x = roll_n(x, axis=dim, n=n_shift)
    return x  # last dim=2 (real&imag)

def roll_n(X, axis, n):

    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]

    return torch.cat([back, front], axis)

class FFT2(nn.Module):
    def __init__(self):
        super(FFT2, self).__init__()

        input_c = 4096
        output_c = 4096

        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.mlp = nn.Sequential(
            nn.Linear(input_c, output_c),
            nn.ReLU(),
            nn.Linear(output_c, output_c),
            nn.ReLU(),
            nn.Linear(output_c, output_c),
        )

    def forward(self, input): # input [B, C, H, W]
        B,C,_,_ = input.shape
        # # print(input.shape)
        median_output = torch.rfft(input, 2, onesided=False) 
        median_output_r = median_output[:, :, :, :, 0]
        median_output_i = median_output[:, :, :, :, 1]

        output = torch.sqrt(median_output_r ** 2 + median_output_i ** 2 + 1e-15) 
        out_put = fftshift2d(output) # 

        pool_fea = self.pool(out_put).view(B,C,-1)
        out_put = F.normalize(pool_fea, p=2, dim=2)
        out_put = out_put.view(B,-1)
        out_put = F.normalize(out_put, p=2, dim=1)

        return out_put # [B, C, H, W]



