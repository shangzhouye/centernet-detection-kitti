import os
import sys
import torch
import numpy as np
from loss import CtdetLoss
from torch.utils.data import DataLoader
from dataset import ctDataset
from DLAnet import DlaNet

def main():

    use_gpu = torch.cuda.is_available()
    print("Use CUDA? ", use_gpu)

    model = DlaNet(34)

    if (use_gpu):
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
        print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

    loss_weight = {'hm_weight':1, 'wh_weight':0.1, 'reg_weight':0.1}
    criterion = CtdetLoss(loss_weight)

    if use_gpu:
        device = torch.device("cuda")
        model.cuda()
    else:
        device = torch.device("cpu")
    
    model.train()

    learning_rate = 5e-4
    num_epochs = 70

    # different learning rate
    params=[]
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        params += [{'params':[value], 'lr':learning_rate}]

    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-4)
    
    # split into training and testing set
    full_dataset = ctDataset()
    full_dataset_len = full_dataset.__len__()
    print("Full dataset has ",  full_dataset_len, " images.")
    train_size = int(0.8 * full_dataset_len)
    test_size = full_dataset_len - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], \
                                                                generator=torch.Generator().manual_seed(42))
    print("Training set and testing set has: ", train_dataset.__len__(), \
            " and ", test_dataset.__len__(), " images respectively.")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    best_test_loss = np.inf 

    loss_log = np.empty((0, 3))

    for epoch in range(num_epochs):
        model.train()
        if epoch == 45:
            learning_rate = learning_rate * 0.1 
        if epoch == 60:
            learning_rate = learning_rate * (0.1 ** 2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        total_loss = 0.0

        for i, sample in enumerate(train_loader):

            for k in sample:
                sample[k] = sample[k].to(device=device, non_blocking=True)

            pred = model(sample['input'])
            loss = criterion(pred, sample)    
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 5 == 0:

                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
                %(epoch+1, num_epochs, i+1, len(train_loader), loss.data, total_loss / (i+1)))

        # validation
        validation_loss = 0.0
        model.eval()
        for i, sample in enumerate(test_loader):
            if use_gpu:
                for k in sample:
                    sample[k] = sample[k].to(device=device, non_blocking=True)
            
            pred = model(sample['input'])
            loss = criterion(pred, sample)   
            validation_loss += loss.item()
        validation_loss /= len(test_loader)
        
        print('Epoch [%d/%d] Validation loss %.5f' % (epoch+1, num_epochs, validation_loss))

        loss_log = np.append(loss_log, [[epoch+1, total_loss / len(train_loader), validation_loss]], axis=0)
        np.savetxt('../loss_log.csv', loss_log, delimiter=',')

        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('Get best test loss.')
            torch.save(model.state_dict(),'../best.pth')
        
        torch.save(model.state_dict(),'../' + str(epoch+1) + '_epoch.pth')
            

if __name__ == "__main__":
    main()