"""This is my model of Fast Nested UNet3D"""
"""Adding the part of predictions"""
import torch
import torch.nn as nn
import time
import numpy as np
# from thop import profile


class VesSAP(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(VesSAP,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv3d(in_ch,5,kernel_size=3, stride=1, padding=1,groups=1),
            nn.BatchNorm3d(5),
            nn.ReLU(inplace=True),

            nn.Conv3d(5, 10, kernel_size=5, stride=1, padding=2, groups=1),
            nn.BatchNorm3d(10),
            nn.ReLU(inplace=True),

            nn.Conv3d(10, 20, kernel_size=5, stride=1, padding=2, groups=1),
            nn.BatchNorm3d(20),
            nn.ReLU(inplace=True),

            nn.Conv3d(20, 50, kernel_size=3, stride=1, padding=1, groups=1),
            nn.BatchNorm3d(50),
            nn.ReLU(inplace=True),

            nn.Conv3d(50, 1, kernel_size=1, stride=1, padding=0, groups=1),
        )

    def forward(self, input):
        xout1 = self.conv(input)

        return xout1




if __name__=="__main__":
    device=torch.device("cuda")
    img=torch.rand(1,1,40,40,40).to(device)
    model=VesSAP(1,1)
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

    # flops,params=profile(model,inputs=(1,1,96,96,96))
    # print('flops=',str(flops/1e6),'M')
    # print('params=',str(params/1e6),'M')









