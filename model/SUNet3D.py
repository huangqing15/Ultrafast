import torch
import torch.nn as nn


def conv_block_3d(in_dim, out_dim, d1,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        act_fn,
        nn.Dropout3d(d1),
    )
    return model

def conv_block_2_3d(in_dim, out_dim, d1,act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim, out_dim, d1,act_fn),
        conv_block_3d(out_dim, out_dim, d1,act_fn),

    )
    return model


def conv_trans_block_3d(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        act_fn,
    )
    return model


def maxpool_3d():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool




class SUNet3D(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(SUNet3D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = nn.ELU(inplace=True)

        print("\n------Initiating U-Net------\n")

        self.down_1 = conv_block_3d(self.in_dim,4,0.1, act_fn)
        self.pool_1 = maxpool_3d()

        self.down_2 = conv_block_3d(4, 8, 0.1, act_fn)
        self.pool_2 = maxpool_3d()

        self.bridge = conv_block_2_3d(8,16,0.2, act_fn)

        self.trans_1 = conv_trans_block_3d(16, 8, act_fn)
        self.up_1 = conv_block_3d(8, 8, 0.1, act_fn)

        self.trans_2 = conv_trans_block_3d(8,4, act_fn)
        self.up_2 = conv_block_3d(8, 4, 0.1, act_fn)

        self.out = nn.Conv3d(4, out_dim, kernel_size=1)

    def forward(self, x):
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        bridge = self.bridge(pool_2)

        trans_1 = self.trans_1(bridge)
        up_1 = self.up_1(trans_1)

        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_1], dim=1)
        up_2 = self.up_2(concat_2)

        out = self.out(up_2)

        return out


if __name__ == '__main__':
    # Unet_3D=UnetGenerator_3d(1,2,8)
    # print(Unet_3D)

    device = torch.device("cuda")
    img = torch.rand(1, 1, 40, 40, 40).to(device)

    model = SUNet3D(1, 1)
    model = model.to(device)

    output1 = model(img)

    # print(output1.size())

    print(model)

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
