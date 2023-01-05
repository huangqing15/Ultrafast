from utils.show_results import Read_Tiff
from Common import *
from GetModel import *
from utils.tools import *
from utils.meters import AverageMeter
import torch.nn as nn
import torch
import time
from utils.eval_metric import ConfusionMeter
import os
import numpy as np
from dataset.generatePatches import *
from torch.utils.data import DataLoader
from utils.show_results import Get_Compared_MIP,Get_Joint_MIP,ImageForVis
from utils.metrics_xu import batch_soft_metric
import skimage.io as skio
from glob import glob
from dataset.tools import WriteList2Txt,ReadTxt2List
import math
import h5py
from utils.tools import list2csv
from GetModel import *

'''This uses the calculation whether the data should be convert to cpu'''

all_mean1 = np.array([168.5], dtype=np.float32)
all_std1 = np.array([500], dtype=np.float32)


def Normalization(img):
    img=np.array(img,np.float)
    return (img-np.min(img))/(np.max(img)-np.min(img))


def calculate_fit_patchsize(image_size,overlap):
    #### to fast the processing procedure
    patch_size_chosen = [ 96, 104, 112, 120, 128]
    # patch_size_chosen = [64]

    ### chosen the right patch size for each dimension
    cal_lenx = [math.ceil((image_size[0] - overlap[0]) / (ii - overlap[0])) * ii for ii in patch_size_chosen]
    cal_leny = [math.ceil((image_size[1] - overlap[1]) / (ii - overlap[1])) * ii for ii in patch_size_chosen]
    cal_lenz = [math.ceil((image_size[2] - overlap[2]) / (ii - overlap[2])) * ii for ii in patch_size_chosen]

    # print(cal_lenx)

    patch_size = [patch_size_chosen[np.argmin(np.array(cal_lenx))], patch_size_chosen[np.argmin(np.array(cal_leny))],
                  patch_size_chosen[np.argmin(np.array(cal_lenz))]]


    return patch_size


def calculate_fit_index(image_sizei,patch_sizei,patch_num_dimi,overlapi):
    xx1 = [(patch_sizei - overlapi) * ii for ii in range(patch_num_dimi)]
    xx1[-1] = image_sizei - patch_sizei
    return xx1


def calculate_patch_index(image_size,patch_size,overlap):
    patch_num_dim = [math.ceil((image_size[0] - overlap[0]) / (patch_size[0] - overlap[0])),
                     math.ceil((image_size[1] - overlap[1]) / (patch_size[1] - overlap[1])),
                     math.ceil((image_size[2] - overlap[2]) / (patch_size[2] - overlap[2])),
                     ]

    # patch_num = patch_num_dim[0] * patch_num_dim[1] * patch_num_dim[2]

    xx1 = calculate_fit_index(image_size[0], patch_size[0], patch_num_dim[0], overlap[0])
    yy1 = calculate_fit_index(image_size[1], patch_size[1], patch_num_dim[1], overlap[1])
    zz1 = calculate_fit_index(image_size[2], patch_size[2], patch_num_dim[2], overlap[2])

    # get the patch indexes of the image size
    patch_index_start = [(aa, bb, cc) for aa in xx1 for bb in yy1 for cc in zz1]

    return patch_index_start


def get_3d_patch(image,  patch_index, patch_size):
    patch_index = np.asarray(patch_index, dtype=np.int16)
    patch_size = np.asarray(patch_size)

    return image[..., patch_index[0]:patch_index[0] + patch_size[0], patch_index[1]:patch_index[1] + patch_size[1],
           patch_index[2]:patch_index[2] + patch_size[2]]


def get_a_patch_index(patch_index,patch_size):
    x0,y0,z0=patch_index
    x1,y1,z1=x0+patch_size[0],y0+patch_size[1],z0+patch_size[2]
    return x0,y0,z0,x1,y1,z1


def reconstruct_from_patches(patches, patch_indices, data_shape):
    data = np.zeros(data_shape)

    for patch, index in zip(patches, patch_indices):
        x0, y0, z0 = index[0], index[1], index[2]
        x1, y1, z1 = index[0] + patch.shape[-3], index[1] + patch.shape[-2], index[2] + patch.shape[-1]
        # calculate the new area
        data[x0:x1, y0:y1, z0:z1]= patch
    return data


def Get_Combined_MIP(img, pred):

    im0 = np.max(img, 0)
    im0[im0>250]=250
    im0[im0<100]=100
    im0=255*(im0-100)/150
    im0=np.array(im0,np.uint8)

    im1 = np.max(pred, 0)
    out = np.vstack([im0, im1])
    return out

def Read_3D_Tiff(image_name):
    image1=tifffile.imread(image_name)
    return image1

class My_Sigmoid(nn.Module):
    def __init__(self):
        super(My_Sigmoid, self).__init__()
        self.model1 = nn.Sigmoid()
        self.model1= nn.DataParallel(self.model1).cuda()

    def forward(self, input):
        return self.model1(input)



##########################################################################################

class GenerateTestDataset_Joint():
    def __init__(self,image, patch_size,patch_index_start):
        self.patches_num = len(patch_index_start)
        self.patch_size=patch_size
        self.patchindices = patch_index_start

        self.image=image.astype(np.float32)

        ## normalizae
        self.image = (self.image - all_mean1) / all_std1

    def __len__(self):
        return self.patches_num

    def __getitem__(self, ind ):
        # get the patches of the image and label
        image_patch = get_3d_patch(self.image, self.patchindices[ind],self.patch_size)

        # To expand the dim of the dataset and turn the dataset into torch
        image_patch=np.expand_dims(image_patch,axis=0)
        image_patch=torch.from_numpy(image_patch).float()

        return image_patch


def test_model_Joint(image_list, image_num,overlap,model,save_3d_path,save_2d_path):
    # get the dataset for testing
    image_name = image_list[image_num]
    image=Read_3D_Tiff(image_name)
    print(image_name)

    #### 1. start to calculate the index for all the patches
    image_size=image.shape
    patch_size = calculate_fit_patchsize(image_size, overlap)
    # patch_size=[64,64,64]
    # print(patch_size)

    ### 2. calculate the start and end patch size
    patch_index_start = calculate_patch_index(image_size, patch_size, overlap)

    ### 3. begin to build the final probability prediction for the prob
    prob_out=np.zeros(image_size,np.uint8)

    # get the dataset
    Tdataset = GenerateTestDataset_Joint(image, patch_size, patch_index_start)
    bt_size=6
    val_loader = DataLoader(Tdataset, batch_size=bt_size, num_workers=4, shuffle=False)

    # begin to test
    model.eval()
    sigm1=My_Sigmoid()

    all_time=0
    for batch_ids, (image_patch) in enumerate(val_loader):
        if opt.use_cuda:
            image_patch=image_patch.cuda()

        with torch.no_grad():
            torch.cuda.synchronize()
            st11 = time.time()
            output3 = model(image_patch)
            prob_patch=sigm1(output3)
            del image_patch,output3
            torch.cuda.synchronize()
            ed11 = time.time()
            # print('GPU time for the model is :', ed11 - st11)
            all_time=all_time+ed11-st11


            prob_patch=255*prob_patch[:, 0, ...]
            prob_patch = prob_patch.byte()
            # calculate the idx
            patch_id11=bt_size*batch_ids+np.arange(prob_patch.shape[0])
            prob_judge=prob_patch>0
            prob_judge1=[torch.sum(prob_judge[id,...]) for id in range(prob_patch.shape[0])]
            prob_judge1 = np.array(prob_judge1)
            del prob_judge

            # delete the idx where the value <100
            keep_idx=[]
            for id in range(prob_patch.shape[0]):
                if prob_judge1[id]>200:
                    keep_idx.append(id)
            # print(prob_judge1,keep_idx)

            if keep_idx:
                keep_idx=np.array(keep_idx,int)
                patch_id11=patch_id11[keep_idx]
                # print('keep_idx:{} patch_id11:{}'.format(keep_idx,patch_id11))

                prob_patch=prob_patch[keep_idx,...]
                prob_patch = prob_patch.data.cpu().numpy()
                # print('keep_idx:{} patch_id11:{} prob_patch shape:{}'.format(keep_idx, patch_id11,prob_patch.shape))

                for id in range(prob_patch.shape[0]):
                    patch_num=patch_id11[id]
                    x0, y0, z0, x1, y1, z1 = get_a_patch_index(patch_index_start[patch_num], patch_size)
                    prob_out[x0:x1, y0:y1, z0:z1] = prob_patch[id, :, :, :]
            del prob_patch

    # print('the GPU model time: ', all_time)


    image_name1=image_list[image_num]
    image_name1=image_name1.split('/')[-1].split('.')[0]

    st11=time.time()
    tifffile.imsave(os.path.join(save_3d_path, '{}_prob.tif'.format(image_name1)), prob_out)
    ed11=time.time()
    # print('save tifffile time ', ed11-st11)

    st1=time.time()
    image_mip = Get_Combined_MIP(image, prob_out)
    mip_name = os.path.join(save_2d_path,'{}_mip.png'.format(image_name1))
    skio.imsave(mip_name, np.uint8(image_mip))
    ed1=time.time()
    # print('mip save time is ',ed1-st1)
    return all_time

##########################################################################################




##########################################################################################
class GenerateTestDataset_Direct():
    def __init__(self,image):
        self.image=image.astype(np.float32)
        self.image = (self.image - all_mean1) / all_std1

    def __len__(self):
        return 1

    def __getitem__(self, ind ):
        # To expand the dim of the dataset and turn the dataset into torch
        image_patch=np.expand_dims(self.image,axis=0)
        image_patch=torch.from_numpy(image_patch).float()
        return image_patch


def test_model_Direct(image_list, image_num, model,save_3d_path,save_2d_path):
    # get the dataset for testing
    image_name = image_list[image_num]
    image=Read_3D_Tiff(image_name)
    # image = image[0:96, 0:96, 0:96]
    image = image[96:192, 96:192, 96:192]

    # get the dataset
    Tdataset = GenerateTestDataset_Direct(image)
    val_loader = DataLoader(Tdataset, batch_size=1, num_workers=4, shuffle=False)

    # begin to test
    model.eval()
    sigm1=My_Sigmoid()

    for batch_ids, (image_patch) in enumerate(val_loader):
        if opt.use_cuda:
            image_patch=image_patch.cuda()

        torch.cuda.synchronize()
        st1=time.time()
        with torch.no_grad():
            # _,_,output3= model(image_patch)

            output3 = model(image_patch)

            prob_out=sigm1(output3)
            prob_out = 255 * prob_out[0,0,...]
            del image_patch,output3
        torch.cuda.synchronize()
        ed1=time.time()
        print('The running time of GPU model is: {}'.format(ed1-st1))

        prob_out=prob_out.cpu().numpy()

    image_name1=image_list[image_num]
    image_name1=image_name1.split('/')[-1].split('.')[0]

    # st11=time.time()
    tifffile.imsave(os.path.join(save_3d_path, '{}_prob.tif'.format(image_name1)), np.uint8(prob_out))
    # ed11=time.time()
    # print('save tifffile time ', ed11-st11)

    # st1=time.time()

    image_mip = Get_Combined_MIP(image, prob_out)
    mip_name = os.path.join(save_2d_path,'{}_mip.png'.format(image_name1))
    skio.imsave(mip_name, np.uint8(image_mip))
    # ed1=time.time()
    # print('mip save time is ',ed1-st1)
    return ed1-st1

##########################################################################################





if __name__=="__main__":
        parameters_name = '/home/hp/Neuro_Separate/CMP_Data_Fast/checkpoints/Fast_MyNet5_Re300_epoch_24.ckpt'
        model = GetModel(opt)
        model_CKPT = torch.load(parameters_name)
        model.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!')

        file_path = '/media/hp/work/DB_Cmp_Data/image1'
        image_list = glob(os.path.join(file_path, '*.tif'))
        image_list=[os.path.join(file_path, i) for i in image_list]


        # begin to evaluate the dataset
        data_name = 'FMyNet5_BigView_NoSup_DB_Cmp'
        save_3d_path = '/media/hp/work/DB_Cmp_Data/' + data_name
        save_2d_path = '/media/hp/work/DB_Cmp_Data/mip_' + data_name
        remake_dirs(save_3d_path)
        remake_dirs(save_2d_path)

        # record the list name
        list_name = '/media/hp/work/DB_Cmp_Data/' + data_name + '.csv'
        result_list = ['name_id', 'time_model','time_all']
        list2csv(result_list, list_name, mode='w')

        overlap = [0, 0, 0]
        image_len = len(image_list)
        for image_num in range(image_len):
            st11=time.time()
            time_model =test_model_Joint(image_list, image_num, overlap, model, save_3d_path,save_2d_path)
            # time_model=test_model_Direct(image_list, image_num, model, save_3d_path, save_2d_path)
            ed11=time.time()
            result_list = [image_num, time_model,ed11-st11]
            list2csv(result_list, list_name)

            print('image_num: {}, image_name: {}, one model time:{}, one running time:{}'.format(image_num,image_list[image_num],time_model, ed11 - st11))
































