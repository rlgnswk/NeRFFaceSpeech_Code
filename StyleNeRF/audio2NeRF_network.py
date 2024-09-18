import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import persistence
from torch_utils.ops import bias_act

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

class inpaintDecoder(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.hidden_dim = 128

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, n_features)
        )
        
    def forward(self, x):

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)

        return x

class seg_Decoder(torch.nn.Module):
    def __init__(self, in_feature, out_feature, mid_feature=256):
        super().__init__()
        self.hidden_dim = mid_feature

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(in_feature, self.hidden_dim),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, out_feature)
        )
        
    def forward(self, x):

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)

        return x

class transfer_decoder_ver2(torch.nn.Module):
    def __init__(self, in_feature=64 + 512, out_feature=512, mid_feature=512):
        super().__init__()
        self.hidden_dim = mid_feature

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(in_feature, self.hidden_dim),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, out_feature)
        )
        
    def forward(self, w, exp):
        w_in = w
        
        batch_size = w.shape[0]
        concat_in = torch.cat((w,exp), dim=-1) # [B, w + exp)]
        
        out = self.net(concat_in)
        
        return w_in + out   
    

class transfer_decoder(nn.Module):
    def __init__(self, input_size= 64 + 512, hidden_size = 512, output_size=512):
        super(transfer_decoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size , hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size) 
        
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, w, exp):
        # reshape
        #import ipdb; ipdb.set_trace()
        w_in = w
        
        batch_size = w.shape[0]

        #import ipdb; ipdb.set_trace()
        concat_in = torch.cat((w,exp), dim=-1) # [B, w + exp)]
        #print(concat_in.shape)
        #import ipdb; ipdb.set_trace()
        out = self.fc1(concat_in)
        out = F.relu(out)
        out = self.fc2(out)

        out = F.relu(out)
        out = self.fc3(out)

        out = F.relu(out)
        out = self.fc4(out)
        
        #import ipdb; ipdb.set_trace()
        return w_in + out
    
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 인코더 부분
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 디코더 부분
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        )

    def forward(self, feat, mask, size=224):
        # 인코더 부분
        feat = torch.nn.functional.interpolate(feat, size=(size,size), mode='bicubic')
        
        masked_feat = feat * mask#.repeat(1, feat.size(1), 1, 1)
        out = self.encoder(masked_feat)
        
        # 디코더 부분
        out = self.decoder(out)
        out = out + masked_feat
        
        out = torch.nn.functional.interpolate(out, size=(32,32), mode='bicubic')
        
        return out

class UNet_v2(nn.Module):
    def __init__(self):
        super(UNet_v2, self).__init__()

        # 인코더 부분
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        
        self.bottleNeck = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        
        # 디코더 부분
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        )

    def forward(self, feat, mask, size=224, Maskmode="nearest", Featmode="bicubic"):
        # 인코더 부분
        feat = torch.nn.functional.interpolate(feat, size=(size,size), mode=Featmode)
        mask = torch.nn.functional.interpolate(mask, size=(size,size), mode=Maskmode)
        
        masked_feat = feat * mask#.repeat(1, feat.size(1), 1, 1)
        out = self.encoder(masked_feat)
        
        out = self.bottleNeck(out)

        out = self.decoder(out)
        out = out + masked_feat
        
        out = torch.nn.functional.interpolate(out, size=(32,32), mode=Featmode)
        
        return out


class transfer_decoder_blink(nn.Module):
    def __init__(self, input_size= 68*2 + 512, hidden_size = 512, output_size=512):
        super(transfer_decoder_blink, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size , hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size) 
        
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, w, lm):
        # reshape
        #import ipdb; ipdb.set_trace()
        w_in = w
        
        batch_size = w.shape[0]
        
        lm = lm.view(lm.size(0), -1)
        
        #import ipdb; ipdb.set_trace()
        concat_in = torch.cat((w,lm), dim=-1) # [B, w + exp)]
        #print(concat_in.shape)
        #import ipdb; ipdb.set_trace()
        out = self.fc1(concat_in)
        out = F.relu(out)
        out = self.fc2(out)

        out = F.relu(out)
        out = self.fc3(out)

        out = F.relu(out)
        out = self.fc4(out)
        
        #import ipdb; ipdb.set_trace()
        return w_in + out


    
class UNet_v3(nn.Module):
    def __init__(self):
        super(UNet_v3, self).__init__()

        # 인코더 부분
        self.encoder = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        
        self.bottleNeck = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 디코더 부분
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        )
        
    def forward(self, feat, mask, size=224, Maskmode="nearest", Featmode="bicubic"):
        # 인코더 부분
        
        feat = torch.nn.functional.interpolate(feat, size=(size,size), mode=Featmode) #B, 256, size, size
        mask = torch.nn.functional.interpolate(mask, size=(size,size), mode=Maskmode) #B, 1, size, size
        
        #masked_feat = feat * mask#.repeat(1, feat.size(1), 1, 1)
        cat_feat = torch.cat((feat, mask), dim=1)
        out = self.encoder(cat_feat)
        
        out = self.bottleNeck(out)
        
        out = self.decoder(out)
        out = out*(1.-mask) + feat*mask
        
        out = torch.nn.functional.interpolate(out, size=(32,32), mode=Featmode)
        
        return out
    
    
    
class UNet_v4(nn.Module):
    def __init__(self):
        super(UNet_v4, self).__init__()

        # 인코더 부분
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        
        self.bottleNeck = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 디코더 부분
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        )
        
    def forward(self, feat, mask, size=224, Maskmode="nearest", Featmode="bicubic"):
        # 인코더 부분
        
        feat = torch.nn.functional.interpolate(feat, size=(size,size), mode=Featmode) #B, 256, size, size
        mask = torch.nn.functional.interpolate(mask, size=(size,size), mode=Maskmode) #B, 1, size, size
        
        masked_feat = feat * mask
        #cat_feat = torch.cat((feat, mask), dim=1)
        out = self.encoder(masked_feat)
        
        out = self.bottleNeck(out)
        
        out = self.decoder(out)
        out = out*(1.-mask) + masked_feat
        
        out = torch.nn.functional.interpolate(out, size=(32,32), mode=Featmode)
        
        return out

class UNet_v5(nn.Module):
    def __init__(self):
        super(UNet_v5, self).__init__()

        # 인코더 부분
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        
        self.bottleNeck = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 디코더 부분
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        )
        
    def forward(self, feat, mask, size=224, Maskmode="nearest", Featmode="bicubic"):
        # 인코더 부분
        
        feat = torch.nn.functional.interpolate(feat, size=(size,size), mode=Featmode) #B, 256, size, size
        mask = torch.nn.functional.interpolate(mask, size=(size,size), mode=Maskmode) #B, 1, size, size
        
        masked_feat = feat * mask
        #cat_feat = torch.cat((feat, mask), dim=1)
        out = self.encoder(masked_feat)
        
        out = self.bottleNeck(out)
        
        out = self.decoder(out)
        #out = out*(1.-mask) + masked_feat
        out = out + masked_feat
        
        out = torch.nn.functional.interpolate(out, size=(32,32), mode=Featmode)
        
        return out
    
    

class UNet_VAE(nn.Module):
    def __init__(self):
        super(UNet_VAE, self).__init__()

        # 인코더 부분
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_mean = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.conv_var = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 디코더 부분
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        )
        
    
    def sampling(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean
    
    
    def forward(self, feat, mask, size=224):
        # 인코더 부분
        feat = torch.nn.functional.interpolate(feat, size=(size,size), mode='bicubic')
        
        masked_feat = feat * mask#.repeat(1, feat.size(1), 1, 1)
        
        out = self.encoder(masked_feat)
        conv_mean = self.conv_mean(out)
        conv_var = self.conv_var(out)
        
        out = self.sampling(conv_mean, conv_var)
        
        # 디코더 부분
        out = self.decoder(out)
        
        out = out + masked_feat
        
        out = torch.nn.functional.interpolate(out, size=(32,32), mode='bicubic')

        return out, conv_mean, conv_var