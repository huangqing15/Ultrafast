from Common import *
from GetModel import *
from utils.tools import *
from utils.vis_tool import Visualizer
from config import opt
import torch
import time
from utils.eval_metric import dice_recall_score
from dataset.joint_transform import *
from Validation import val_model
from utils.loss import BCEDiceLoss
from utils.meters import AverageMeter

all_mean1 = np.array([168.5], dtype=np.float32)
all_std1 = np.array([500], dtype=np.float32)



##### used for training:
def TrainOneEpoch(train_loader,model,optimizer,criterion,epoch_num,vis_tool,prefix):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter(),
                  'recall':AverageMeter(),
                  }

    model.train()

    for batch_ids, (data, target) in enumerate(train_loader):
        if opt.use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output1 = model(data)
        # calculate the weight of the batch:
        weight = GetWeight1(opt,target, slr=0.00001,is_t=0)
        loss1 = criterion(output1, target, weight)

        loss1.backward()
        optimizer.step()

        # update the loss value
        dice,recall=dice_recall_score(output1,target)

        # print('dice:{} recall:{}'.format(dice,recall))


        avg_meters['loss'].update(loss1.item(),target.size(0))
        avg_meters['dice'].update(dice,target.size(0))
        avg_meters['recall'].update(recall,target.size(0))

        # # save the parameters
        save_name2 = (prefix + 'current_result')

        state = {'epoch': epoch_num + 1,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 }

        save_parameters(state, save_name2)

        # begin to display
        if batch_ids % opt.train_plotfreq == 0:
            vis_tool.plot('Train_Loss', avg_meters['loss'].avg)
            vis_tool.plot('Train_Dice', avg_meters['dice'].avg)
            vis_tool.plot('Train_Recall', avg_meters['recall'].avg)

            print(avg_meters['loss'].avg,avg_meters['dice'].avg,avg_meters['recall'].avg)

            # begin to plot the prediction result
            # see the image
            image1=data.cpu().numpy()[0,0,...]
            image1 = image1 * all_std1 + all_mean1
            image1=np.clip(image1,150,350)
            # image1=(image1-np.min(image1))/(np.max(image1)-np.min(image1)+1)
            image1 = (image1 - 150) /200
            image1_mip=np.hstack([np.max(image1,0),np.max(image1,1),np.max(image1,2)])

            # see the pred
            # print(output3.shape)
            pred1=torch.sigmoid(output1).data.cpu().numpy()
            # print(pred1.shape)

            pred1=pred1>0.5
            pred1=pred1[0,0,...]
            mip1=np.hstack([np.max(pred1,0),np.max(pred1,1),np.max(pred1,2)])

            # see the label
            target1=target.cpu().numpy()
            target1 =target1[0,0,...]
            mip2 = np.hstack([np.max(target1, 0), np.max(target1, 1), np.max(target1, 2)])
            mip3=np.vstack([image1_mip,mip1,mip2])
            vis_tool.img('pred_label',np.uint8(255*mip3))

        print('Train:Batch_Num:{}  Loss:{:.3f}  Dice:{:.3f}  Recall:{:.3f}'.format(batch_ids,avg_meters['loss'].avg,avg_meters['dice'].avg,avg_meters['recall'].avg))


    return avg_meters['loss'].avg,avg_meters['dice'].avg,avg_meters['recall'].avg


def val_model(opt,val_loader,model,criterion,vis_tool,name1='1'):
    # begin to test the dataset
    model.eval()
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter(),
                  'recall':AverageMeter(),
                  }


    for batch_ids,(data,target) in enumerate(val_loader):
        if opt.use_cuda:
            data,target=data.cuda(),target.cuda()

            # change the output number
            output3=model(data)

            with torch.no_grad():
                loss=criterion(output3,target)
                # pred = torch.sigmoid(output3)
                # pred = pred > 0.5

                # update the loss value
                dice, recall = dice_recall_score(output3, target)
                avg_meters['loss'].update(loss.item(),target.size(0))
                avg_meters['dice'].update(dice,target.size(0))
                avg_meters['recall'].update(recall,target.size(0))

                # begin to play
                if batch_ids % opt.val_plotfreq == 0:
                    vis_tool.plot('Val_Loss'+name1, avg_meters['loss'].avg)
                    vis_tool.plot('Val_Dice'+name1, avg_meters['dice'].avg)
                    vis_tool.plot('Val_Recall'+name1, avg_meters['recall'].avg)

                print('Val: Batch_Num:{}  Loss:{:.3f}  Dice:{:.3f}  Recall:{:.3f}'.format(batch_ids, avg_meters['loss'].avg,avg_meters['dice'].avg,avg_meters['recall'].avg))


    return avg_meters['loss'].avg, avg_meters['dice'].avg, avg_meters['recall'].avg




# define the main function
def main():
    # print the parameters:
    opt._parse()

    train_aug=JointCompose([JointRandomFlip(),
                            JointRandomRotation(),
                            JointRandomGaussianNoise(25,all_std1),
                            JointRandomSubTractGaussianNoise(25,all_std1),
                            JointRandomBrightness([-0.5,0.5]),
                            JointRandomIntensityChange([-100,100],all_std1),
                            ])



    # load the dataloader
    prefix = 'my_rule1_selected_300'

    train_loader = GetDatasetLoader(opt, prefix, phase='train', augument=train_aug)
    val_loader=GetDatasetLoader(opt,prefix,phase= 'val')

    # get the net work
    model=GetModel(opt)

    # if opt.load_state:
    #     ##### fine tune the result
    #     parameters_name='/home/hp/Neuro_Separate/CMP_Data_Dirtribution/checkpoints/rule1_selected_300_epoch_11.ckpt'
    #     model_CKPT = torch.load(parameters_name)
    #     model.load_state_dict(model_CKPT['state_dict'])

    optimizer=GetOptimizer(opt,model)
    scheduler=GetScheduler(opt,optimizer)

    # get the loss function
    criterion=BCEDiceLoss()

    prefix1 = 'Fast_MyNet5_300'
    vis_tool = Visualizer(env=prefix1)
    log_path = 'result'
    log_name = os.path.join(log_path, 'log_{}.txt'.format(prefix1))
    log=WriteLog(log_path,log_name,opt)
    log.write('selected dataset: uses 300 dataset \n')
    log.write('epoch |train_loss |train_dice |train_recall |valid_loss |valid dice |valid_recall |time          \n')
    log.write('------------------------------------------------------------\n')


    # begin to train:
    for epoch_num in range(opt.train_epoch):
    # for epoch_num in np.arange(1,30):
        start_time=time.time()
        scheduler.step()

        avg_loss,train_dice,train_recall=TrainOneEpoch(train_loader, model, optimizer, criterion, epoch_num, vis_tool,prefix1)

        # the information
        run_time = time.time()-start_time
        print('Train Epoch{} run time is:{:.3f}m and {:.3f}s'.format(epoch_num,run_time//60,run_time%60))
        print('Loss:{:.3f}  Recall:{:.3f}  Dice:{:.3f}'.format(avg_loss,train_recall,train_dice))


        if epoch_num>=0:
            save_name2 = (prefix1+'_epoch_{}').format(epoch_num)

            state={'epoch': epoch_num + 1,
                   'state_dict': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   }

            save_parameters(state, save_name2)

        # judge whether to test or not
        if opt.val_run:
            val_avgloss,val_dice,val_recall=val_model(opt,val_loader,model,criterion,vis_tool)

            print('Test Loss:{:.3f}  Recall:{:.3f}  Dice:{:.3f}'.format(val_avgloss,val_recall,val_dice))

        log.write('%d |%0.3f |%0.3f |%0.3f |%0.3f |%0.3f |%0.3f |%0.3f \n' % (epoch_num,avg_loss,train_dice,train_recall,
                                                                    val_avgloss,val_dice,val_recall,run_time))
        log.write('\n')




if __name__=="__main__":
    main()


















































