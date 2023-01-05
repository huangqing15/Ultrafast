from model.UnetGenerator_3d import UnetGenerator_3d as UNet_3D
import torch.nn as nn
import torch
from model.Fast_MyNet5 import Fast_MyNet5
from model.VesSAP import VesSAP
from model.UNet3D_3Layers import UNet3D_3Layers
from model.SUNet3D import SUNet3D

model_choice={
    'UNet_3D':UNet_3D,
    'Fast_MyNet5':Fast_MyNet5,
    'SUNet3D':SUNet3D,
    'VesSAP':VesSAP,
    'UNet3D_3Layers':UNet3D_3Layers,
}


# decide to use which Model: currently UNet
def GetModel(opt):
    if opt.model_choice == 'UNet_3D':
        model=model_choice[opt.model_choice](1,2,opt.num_filter)
    elif opt.model_choice == 'UNet3D_3Layers':
        model = model_choice[opt.model_choice](1, 1, opt.num_filter)
    else:
        # act_fn=nn.LeakyReLU(0.01)
        model = model_choice[opt.model_choice](1, opt.out_dim)


    # decide whether to use cuda or not
    if opt.use_cuda:
        model=nn.DataParallel(model).cuda()

    return model









# if __name__=="__main__":
#     model=GetModel(opt)
#     print(model)

