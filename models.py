import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=True, activation='ReLU'):
        super(ResNetModel, self).__init__()

        self.num_channels = num_channels
        self.resnet = models.resnet50(pretrained=load_weight)
        self.activation = nn.GELU if activation=='GeLU' else nn.ReLU

        for param in self.resnet.parameters():
            param.requires_grad = train_enc
        
        self.conv_layer1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )
        self.conv_layer2 = nn.Sequential(
            self.resnet.maxpool,
            self.resnet.layer1
        )
        self.conv_layer3 = self.resnet.layer2
        self.conv_layer4 = self.resnet.layer3
        self.conv_layer5 = self.resnet.layer4

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1, bias=True),
            self.activation ,
            self.linear_upsampling
        )
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 2048, out_channels = 512, kernel_size = 3, padding = 1, bias = True),
            self.activation,
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            self.activation,
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            self.activation,
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            self.activation,
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
            self.activation,
            nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )
    
    def forward(self, images):
        batch_size = images.size(0)

        out1 = self.conv_layer1(images)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)

        out5 = self.deconv_layer0(out5)
        assert out5.size() == (batch_size, 1024, 16, 16)

        x = torch.cat((out5,out4), 1)
        assert x.size() == (batch_size, 2048, 16, 16)
        x = self.deconv_layer1(x)
        assert x.size() == (batch_size, 512, 32, 32)
        
        x = torch.cat((x, out3), 1)
        assert x.size() == (batch_size, 1024, 32, 32)
        x = self.deconv_layer2(x)
        assert x.size() == (batch_size, 256, 64, 64)

        x = torch.cat((x, out2), 1)
        assert x.size() == (batch_size, 512, 64, 64)
        x = self.deconv_layer3(x)
        assert x.size() == (batch_size, 64, 128, 128)
        
        x = torch.cat((x, out1), 1)
        assert x.size() == (batch_size, 128, 128, 128)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)
        assert x.size() == (batch_size, 1, 256, 256)
        x = x.squeeze(1)
        assert x.size() == (batch_size, 256, 256)
        return x

class Block(nn.Module):
    ''' One block of Unet.
        Contains 2 repeated 3 x 3 unpadded convolutions, each followed by a ReLU.
    '''
    
    def __init__(self, in_channel, out_channel, kernel_size):
        ''' Initialisation '''

        super().__init__()
        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size)
        self.relu   = nn.ReLU()
        
        # Initialise weights on convolutional layers        
        nn.init.normal_(self.conv_1.weight, mean = 0.0, std = self.init_std(in_channel, kernel_size))
        nn.init.normal_(self.conv_1.weight, mean = 0.0, std = self.init_std(out_channel, kernel_size))
    
    
    @staticmethod
    def init_std(channels, kernel_size):
        ''' Computes std for weight initialisation on the convolutional layers'''
        return 2.0 / np.sqrt(channels * kernel_size ** 2)
    
        
    def forward(self, x):
        ''' Forward Phase '''
        
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        
        return x

class UNetResNet50(nn.Module):
    def __init__(self, num_classes=1, train_enc=False):
        super(UNetResNet50, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        self.resnet = resnet

        for param in self.resnet.parameters():
            param.requires_grad = train_enc
        
        # Encoder: take layers from ResNet50
        self.encoder_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4
        
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)

        self.decoder_block4 = Block(2048, 1024, 1024)
        self.decoder_block3 = Block(1024, 512, 512)
        self.decoder_block2 = Block(512, 256, 256)
        self.decoder_block1 = Block(256, 64, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        x1 = self.encoder_conv1(x)       
        x2 = self.encoder_layer1(x1)    
        x3 = self.encoder_layer2(x2)    
        x4 = self.encoder_layer3(x3)     
        x5 = self.encoder_layer4(x4)     

        import IPython; IPython.embed()

        d4 = self.upconv4(x5)            
        d4 = torch.cat([d4, x4], dim=1) 
        d4 = self.decoder_block4(d4)     

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.decoder_block3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.decoder_block2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.decoder_block1(d1)

        out = self.final_conv(d1)
        return out