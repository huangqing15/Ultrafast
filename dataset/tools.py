import numpy as np
import scipy.io as io
import random
from config import opt
from utils.tools import *
import sys
from glob import glob
import tifffile

# read the image and label list of corresponding images
def GetImageLabelList_Path(data_dir_root):
    image_path=os.path.join(data_dir_root,'image')
    label_path=os.path.join(data_dir_root,'label')

    # image and label had corresponding name
    name_list = os.listdir(image_path)
    image_list=[os.path.join(image_path,name_num) for name_num in name_list]    # image and label had corresponding name
    label_list=[os.path.join(label_path,name_num) for name_num in name_list]

    return image_list,label_list



def Read_ImageLabelList_PatchesInd(prefix,phase):
    # read the image label list and patches index
    module_path = os.path.dirname(__file__)
    data_path = module_path + '/' + prefix + '/'

    image_list_name = data_path + phase + '_image_list.txt'
    label_list_name = data_path + phase + '_label_list.txt'

    image_list = ReadTxt2List(image_list_name)
    label_list = ReadTxt2List(label_list_name)

    # get the patch index of the dataset
    index_list_name = data_path + phase + '_random_patch_ind.npy'
    index_list = np.load(index_list_name)

    return image_list,label_list,index_list



def GetList(prefix,phase1='train'):
    # generate the list name of the file
    # data_path = './dataset/'
    data_path=prefix+'/'

    image_list_name = data_path + phase1 + '_image_list.txt'
    label_list_name = data_path + phase1 + '_label_list.txt'

    image_list = ReadTxt2List(image_list_name)
    label_list = ReadTxt2List(label_list_name)

    return image_list,label_list


# save the filelist into txt
def WriteList2Txt(name1,ipTable,mode='w'):
    with open(name1,mode=mode) as fileObject:
        for ip in ipTable:
            fileObject.write(ip)
            fileObject.write('\n')


# Read the filelist to list
def ReadTxt2List(name1,mode='r'):
    result=[]
    with open(name1,mode=mode) as f:
        data = f.readlines()   #read all the trsing into data
        for line in data:
            word = line.strip()  # list
            result.append(word)
    return result

# get the random patches
def RandomPatches(image,patch_size):
    w,h,z = image.shape
    pw,ph,pz=patch_size

    # calculate the random patches index
    nw,nh,nz=w-pw,h-ph,z-pz
    iw=random.randint(0,nw-1)
    ih=random.randint(0,nh-1)
    iz=random.randint(0,nz-1)

    # at: different from matlab
    image_patches=image[iw:iw+pw,ih:ih+ph,iz:iz+pz]

    return image_patches,[iw,ih,iz]


def RandomJointPatches(image,label,patch_size):
    w,h,z = image.shape
    pw,ph,pz=patch_size

    # calculate the random patches index
    nw,nh,nz=w-pw,h-ph,z-pz
    iw=random.randint(0,nw-1)
    ih=random.randint(0,nh-1)
    iz=random.randint(0,nz-1)

    # at: different from matlab
    image_patches=image[iw:iw+pw,ih:ih+ph,iz:iz+pz]
    label_patches=label[iw:iw+pw,ih:ih+ph,iz:iz+pz]

    return image_patches,label_patches


def GetPatches(image,patch_size,position):
    pw,ph,pz=patch_size
    iw,ih,iz=position
    image_patches=image[iw:iw+pw,ih:ih+ph,iz:iz+pz]

    return image_patches

# calculate the useful mask numbers
def LabelIndexNum(label_image):
    image1=label_image>0
    image1=image1.flatten()
    sum_index=np.sum(image1)

    return sum_index

def DecideRange(isize,psize,osize):
    i1=np.array(isize,np.int)
    p1=np.array(psize,np.int)
    o1=np.array(osize,np.int)

    index1=np.asarray(np.mgrid[0:i1:(p1-o1)])
    len1=len(index1)
    index1[len1-1]=i1-1-p1

    return index1



# calculate the ordered patch inds
def GetOrderedPatchInds(image_size,patch_size,overlap):
    indx=DecideRange(image_size[0],patch_size[0],overlap[0])
    indy=DecideRange(image_size[1],patch_size[1],overlap[1])
    indz = DecideRange(image_size[2], patch_size[2], overlap[2])

    Xout = [(xx, yy, zz) for xx in indx for yy in indy for zz in indz]

    return Xout


def Normalization(img):
    return (img-np.min(img))/(np.max(img)-np.min(img))


