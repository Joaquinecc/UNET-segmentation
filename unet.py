import torch
import torch.nn as nn

class Downsampling(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Downsampling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)  
        self.relu=nn.LeakyReLU()
        self.maxpol = nn.MaxPool2d(2, stride=2)
    def forward(self, x):
        x = self.relu(self.conv1(x))  
        x = self.relu(self.conv2(x))  
        return self.maxpol(x),x
    
class Upsampling(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Upsampling, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch, kernel_size=3, stride=1, padding=1)  #Receive the concatenation 
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)  
        self.relu=nn.LeakyReLU()

    def forward(self, x,skip):
        x=self.upconv(x)
        #Crop for concatenation
        # diff_x=skip.shape[2] - x.shape[2]
        # diff_y=skip.shape[3] - x.shape[3]
        # skip_cropped = skip[:, :, diff_x//2: -diff_x//2, diff_y//2: -diff_y//2]
        #Concatenate
        x = torch.cat([x, skip], axis=1)
        x = self.relu(self.conv1(x))  
        x = self.relu(self.conv2(x))  
        return x

class Unet(nn.Module):
    def __init__(self,input_channel,class_num):
        super(Unet, self).__init__()
        self.input_channel=input_channel
        self.class_num=class_num
        self.d1= Downsampling(input_channel,64) 
        self.d2= Downsampling(64,128)
        self.d3= Downsampling(128,256)
        self.d4= Downsampling(256,512)

        #Bottleneck
        self.double_conv = nn.Sequential(
                    nn.Conv2d(512, 1024,kernel_size=3, stride=1, padding=1)  ,
                    nn.ReLU(),
                    nn.Conv2d(1024, 1024,kernel_size=3, stride=1, padding=1)  ,
                    nn.ReLU()
                )
        
        self.u1=Upsampling(1024,512)
        self.u2=Upsampling(512,256)
        self.u3=Upsampling(256,128)
        self.u4=Upsampling(128,64)
        
        #HEAD
        self.head = nn.Conv2d(64, self.class_num, kernel_size=1, padding=0)

    def forward(self,x):
        #Downsampling
        x,s1= self.d1(x) 
        x,s2= self.d2(x)
        x,s3= self.d3(x)
        x,s4= self.d4(x) 


        #Bottleneck
        x = self.double_conv(x)

        #Upsampling
        x=self.u1(x,s4)
        x=self.u2(x,s3)
        x=self.u3(x,s2)
        x=self.u4(x,s1)

        #HEAD
        x=self.head(x)
        return x