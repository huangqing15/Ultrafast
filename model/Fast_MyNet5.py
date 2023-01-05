"""This is my model of Fast Nested UNet3D"""
"""Adding the part of predictions"""
import torch
import torch.nn as nn
import time
import numpy as np


class conv3d_3_3(nn.Module):
    """conv---BN----Relu"""
    def __init__(self,in_ch,out_ch,act_fn = nn.LeakyReLU(0.2, inplace=True)):
        super(conv3d_3_3,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1,groups=1),
            nn.BatchNorm3d(out_ch),
            act_fn,
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1,groups=1),
            nn.BatchNorm3d(out_ch),
            act_fn,
        )

    def forward(self, input):
        return self.conv(input)


class conv3d_3_3_3(nn.Module):
    """conv---BN----Relu"""
    def __init__(self,in_ch,out_ch,act_fn = nn.LeakyReLU(0.2, inplace=True)):
        super(conv3d_3_3_3,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(in_ch),
            act_fn,
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            act_fn,
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            act_fn,
        )

    def forward(self, input):
        return self.conv(input)



class down(nn.Module):
    """Max---double_conv3d"""
    def __init__(self,in_ch,out_ch):
        super(down,self).__init__()
        self.conv=nn.Sequential(
            nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
            conv3d_3_3_3(in_ch,out_ch),
        )

    def forward(self, input):
        return self.conv(input)


class trans_conv_3d(nn.Module):
    """trans_conv---BN----Relu"""
    def __init__(self,in_ch,out_ch,ker_size=3,act_fn = nn.LeakyReLU(0.2, inplace=True)):
        super(trans_conv_3d,self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=ker_size, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(out_ch),
            act_fn,)

    def forward(self, input):
        return self.conv(input)



class up1(nn.Module):
    """convup---double_conv2d"""
    def __init__(self,in_ch,out_ch,mid_ch,ker_size=3,act_fn = nn.LeakyReLU(0.2, inplace=True)):
        super(up1,self).__init__()
        self.conv_up =trans_conv_3d(in_ch,out_ch,ker_size,act_fn)
        self.conv=conv3d_3_3(mid_ch,out_ch)

    def forward(self, input1,input2):
        x1=self.conv_up(input1)
        x2=torch.cat([input2,x1],dim=1)
        return self.conv(x2)


class up2(nn.Module):
    """convup---double_conv2d"""
    def __init__(self,in_ch,out_ch,mid_ch,ker_size=3,act_fn = nn.LeakyReLU(0.2, inplace=True)):
        super(up2,self).__init__()
        self.conv_up =trans_conv_3d(in_ch,out_ch,ker_size,act_fn)
        self.conv=conv3d_3_3(mid_ch,out_ch)

    def forward(self, input1,input2,input3):
        x1=self.conv_up(input1)
        x2=torch.cat([input2,input3,x1],dim=1)
        return self.conv(x2)


class upm(nn.Module):
    """convup---double_conv2d"""
    def __init__(self,in_ch,out_ch,ker_size=3,act_fn = nn.LeakyReLU(0.2, inplace=True)):
        super(upm,self).__init__()
        self.conv_up =trans_conv_3d(in_ch,out_ch,ker_size,act_fn)

    def forward(self, input1,input2):
        x1=self.conv_up(input1)
        x2=torch.cat([input2,x1],dim=1)
        return x2

class out_conv(nn.Module):
    def __init__(self,in_ch,out_ch,ker_size=1):
        super(out_conv,self).__init__()
        self.conv=nn.Conv3d(in_ch,out_ch,kernel_size=ker_size)

    def forward(self, input):
        return self.conv(input)


class Fast_MyNet5(nn.Module):
    def __init__(self,in_ch,out_ch,st_filter=8):
        super(Fast_MyNet5,self).__init__()
        self.inc=conv3d_3_3(in_ch,st_filter)

        self.x10=down(st_filter,2*(st_filter-1))
        self.x20=down(2*(st_filter-1),4*(st_filter-1))

        self.x11=up1(4*(st_filter-1),2*(st_filter-1),4*(st_filter-1))
        self.x01 =upm(2*(st_filter-1),st_filter)
        self.x02=up1(2*(st_filter-1),st_filter,st_filter*3)

        self.clc1 = out_conv(st_filter, out_ch)
        self.clc2 = out_conv(st_filter*2, out_ch)
        self.clc3 = out_conv(st_filter, out_ch)

    def forward(self, input):
        x00=self.inc(input)
        xout1 = self.clc1(x00)

        x10=self.x10(x00)
        x01 = self.x01(x10, x00)
        xout2 = self.clc2(x01)

        x20=self.x20(x10)
        x11=self.x11(x20,x10)
        x02=self.x02(x11,x01)

        xout3 = self.clc3(x02)
        xout=xout2+xout1+xout3

        return xout




if __name__=="__main__":
    device=torch.device("cuda")
    img=torch.rand(1,1,40,40,40).to(device)
    model=Fast_MyNet5(1,1)
    model=model.to(device)
    output1=model(img)

    # print(model)

    st1=time.time()
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l

    print("总参数数量和：" + str(k/1e6))
    ed1=time.time()
    run1=ed1-st1
    print(run1)









