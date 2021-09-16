import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
os.sys.path.append('/Users/dj/Downloads/Handover')

from config import config
from dataloader import VideoDataset
from network import AR3D


os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#cpu gpu 확인
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using {} device'.format(device))

if config.resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(config.save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(config.save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(config.save_dir_root, 'run', 'run_'+str(run_id))
save_name = config.model_name + '_' + config.dataset + '_' + config.sfe_type + '_' +config.ar3d_version

def train_model(datset = config.dataset, save_dir = save_dir, num_classes = config.num_classes, lr = config.lr,
                num_epochs = config.nEpochs, save_epoch = config.snapshot, useTest = config.useTest,
                test_interval = config.nTestInterval, pretrained = config.pretrained):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    

    if config.model_name == 'AR3D':
        model = AR3D.AR3D(num_classes = num_classes, AR3D_V = config.ar3d_version, SFE_type = config.sfe_type, 
                          attention_method = config.attention, reduction_ratio = config.reduction_ratio, hidden_unit = config.hidden_units)
               
        train_params = [{'params': AR3D.get_1x_lr_params(model), 'lr': lr},
                        # {'params': AR3D.get_10x_lr_params(model), 'lr': lr}]
                        {'params': AR3D.get_10x_lr_params(model), 'lr': lr * 10}]
        
    else:
        print('Other models are not implemented yet...')
        raise NotImplementedError
    
    criterion = nn.CrossEntropyLoss()

    if len(config.cuda_devices) == 1:
        model.to(device)
        criterion.to(device)
    else:
        model = nn.DataParallel(model, device_ids=config.cuda_devices)
        # model.cuda()
        criterion.to(device)
        
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(train_params, lr=config.lr, momentum=config.momentum, weight_decay= config.wd)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step, gamma=config.gamma)  
        # the scheduler divides the lr by 10 every 10 epochs
    
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(train_params, lr=config.lr, betas=(0.9, 0.999), eps=config.eps, weight_decay=config.wd)
        # default settings for Adam eps=1e-8, wd=0
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

    if pretrained:
        if config.resume_epoch == 0: 
        ## without resume epoch, manually set pretrained model path
            ckpt = torch.load(config.pretrained_model)
        else:
            ckpt = torch.load(os.path.join(save_dir, 'models', save_name + '_epoch-' + str(config.resume_epoch - 1) + '.pth.tar'),
                        map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
            print("Initializing weights from: {}...".format(
                os.path.join(save_dir, 'models', save_name + '_epoch-' + str(config.resume_epoch - 1) + '.pth.tar')))
        optimizer.load_state_dict(ckpt['opt_dict'])
        model.load_state_dict(ckpt['state_dict'])
    else:
        print("Training {} from scratch...".format(config.model_name))
    
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(config.dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=config.dataset, split='train',clip_len=config.frames_per_clips), batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader   = DataLoader(VideoDataset(dataset=config.dataset, split='val',  clip_len=config.frames_per_clips), batch_size=config.batch_size, num_workers=config.num_workers)
    if config.useTest == True:
        test_dataloader  = DataLoader(VideoDataset(dataset=config.dataset, split='test', clip_len=config.frames_per_clips), batch_size=config.batch_size, num_workers=config.num_workers)
        test_size = len(test_dataloader.dataset)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    
    with open(config.train_log, "a") as train_log:
        train_log.write('Called with configuration....\n')
    for k in config:
        with open(config.train_log, "a") as train_log:
                train_log.write(str(k) + '\t' + str(config[k]) + '\n')

    for epoch in range(config.resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            iter_count = 0
            for inputs, labels in tqdm(trainval_loaders[phase]):
                #print(inputs)
                # if phase == 'train':
                #     # scheduler.step() is to be called once every epoch during training
                #     scheduler.step()
                
                # move inputs and labels to the device the training is taking place on

                # WARNING:root:NaN or Inf found in input tensor.
                # check nan or inf in input tensor
                # if torch.sum(torch.isnan(inputs)) != torch.tensor(0):
                #     print('NaN in input tensor')
                # if torch.sum(torch.isinf(inputs)) != torch.tensor(0):
                #     print('Inf in input tensor'
            
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)

                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)

                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                labels=labels.long()
                # loss = criterion(outputs, labels)
                loss = criterion(outputs+1e-8, labels)

                if phase == 'train':
                    # at the end of each batch_iter
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if iter_count%config.logging_step == 0:
                    with open(config.train_log, "a") as train_log:
                        train_log.write("[{}] Epoch iter_{}: {}/{} Loss: {} Acc: {} \n".format(phase, iter_count+1, epoch+1, config.nEpochs, running_loss/((iter_count+1)*config.batch_size), running_corrects.double()/((iter_count+1)*config.batch_size)))
                iter_count += 1

            if phase == 'train':
                scheduler.step()
                # scheduler.step() is to be called once every epoch during training
            
            # TODO: replace code below
            # epoch_loss should not devide by trainval_size[phase](~= 230000 for kinetics400)
            # should devide by total_iter(~=230000/batch_size)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)
            with open(config.train_log, "a") as train_log:
                        train_log.write("[{}] Epoch: {}/{} Loss: {} Acc: {} \n".format(phase, epoch+1, config.nEpochs, epoch_loss, epoch_acc))
            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, config.nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

            if phase == 'train':
                    # scheduler.step() is to be called once every epoch during training
                    scheduler.step()

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', save_name + '_epoch-' + str(epoch) + '.pth.tar'))
            with open(config.train_log, "a") as train_log:
                train_log.write("Save model at {}\n".format(os.path.join(save_dir, 'models', save_name + '_epoch-' + str(epoch) + '.pth.tar')))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', save_name + '_epoch-' + str(epoch) + '.pth.tar')))
        
        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, config.nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
        
    writer.close()
    
if __name__ == "__main__":
    train_model()

