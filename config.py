from pprint import pprint
import torch
import numpy as np
import os
import time

class Config:
    ##############: for data prepare: no resize operation
    patch_size = [120, 120, 120]
    pixel_size=[0.2,0.2,1]

    ######################## for dataset
    dataset_prefix='rule_300'

    # ##############: for training
    train_augument=True
    limited_range=[0,1500]
    train_shuffile=True
    train_batch = 3
    num_workers = 5
    use_cuda=True

    # model_choice = 'UNet_3D'
    model_choice = 'Fast_MyNet5'
    # model_choice = 'VesSAP'
    # model_choice='SUNet3D'

    in_dim=1
    out_dim=1
    num_filter=32

    current_state='train'
    # current_state = 'test'
    # save_parameters_name = 'Fast_MyNet5'
    save_parameters_name = 'SUNet_300'

    load_state=True

    # record the result
    log_path='result'
    log_name = os.path.join(log_path, 'log_{}.txt'.format(save_parameters_name))

    # optimizer
    optimizer='SGD'
    lr=0.01
    momentum=0.9
    weight_decay=0.0003

    scheduler='StepLR'
    step_size=4
    gamma=0.5

    # train parameters
    train_epoch = 80
    train_plotfreq=2

    val_run=True
    # val_run = False
    val_plotfreq=2
    save_img=True

    ###################: for validatation
    val_batch=3
    test_batch=3

    ###################: for test
    recon_dir_root='/media/hp/work/result/'
    recon_mip_dir_root=recon_dir_root+'_mip'
    image_size = [300, 300, 300]
    image_num = 100
    overlap = 10
    patch_valnum = np.ceil(image_size[0] / (patch_size[0] - overlap)) ** 3
    thred=32

    ###############: print the informations  ###############
    def _parse(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}



###### build the instance
opt = Config()

