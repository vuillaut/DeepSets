import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm, trange
import numpy as np
from datetime import datetime

import classifier
import modelnet
# from pytorch_lightning import Trainer
# from pytorch_lightning import loggers as pl_loggers

from torch.utils.tensorboard import SummaryWriter
from deepcompton.utils import angular_separation

#################### Settings ##############################
num_epochs = 2000
batch_size = 4
downsample = 20    #For 5000 points use 2, for 1000 use 10, for 100 use 100
network_dim = 512  #For 5000 points use 512, for 1000 use 256, for 100 use 256
num_repeats = 1    #Number of times to repeat the experiment
data_path = '/uds_data/glearn/workspaces/thomas/astroinfo21/Compton/Data/gold_angles.h5'
log_dir = f'/uds_data/glearn/workspaces/thomas/astroinfo21/Compton/experiments/run/{datetime.now()}'
cuda = True
#################### Settings ##############################

# tb_logger = pl_loggers.TensorBoardLogger("logs/")
writer = SummaryWriter(log_dir=log_dir)


class PointCloudTrainer(object):
    def __init__(self):
        #Data loader
        self.model_fetcher = modelnet.ModelFetcher(data_path, batch_size, downsample, do_standardize=True, do_augmentation=True)

        #Setup network
        if cuda:
            self.D = classifier.DTanhCompton(network_dim, pool='max1').cuda()
            # self.L = nn.CrossEntropyLoss().cuda()
            # self.L = nn.MSELoss().cuda()
            self.L = classifier.CosAngularSepLoss().cuda()
        else:
            self.D = classifier.DTanhCompton(network_dim, pool='max1')
            # self.L = nn.CrossEntropyLoss()
            # self.L = nn.MSELoss()
            self.L = classifier.CosAngularSepLoss()
        self.optimizer = optim.Adam([{'params':self.D.parameters()}], lr=1e-3, weight_decay=1e-7, eps=1e-3)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=list(range(400,num_epochs,400)), gamma=0.1)
        #self.optimizer = optim.Adamax([{'params':self.D.parameters()}], lr=5e-4, weight_decay=1e-7, eps=1e-3) # optionally use this for 5000 points case, but adam with scheduler also works

    def train(self):
        self.D.train()
        loss_val = float('inf')
        global_count = 0
        for j in trange(num_epochs, desc="Epochs: "):
            counts = 0
            sum_as = 0.0
            train_data = self.model_fetcher.train_data(loss_val)
            for ii, (x, _, y) in enumerate(train_data):
                global_count += 1
                counts += len(y)
                if cuda:
                    X = Variable(torch.cuda.FloatTensor(x))
                    Y = Variable(torch.cuda.FloatTensor(y))
                else:
                    X = Variable(torch.FloatTensor(x))
                    Y = Variable(torch.FloatTensor(y))
                self.optimizer.zero_grad()
                f_X = self.D(X)
                # loss = self.L(f_X, Y)
                loss = self.L(f_X[:,0], f_X[:,1], Y[:,0], Y[:,1])

                loss_val = loss.data.cpu().numpy()[()]
                # sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()[()]
                sum_as += angular_separation(f_X[:,0].detach().cpu().numpy(), f_X[:,1].detach().cpu().numpy(),
                                             Y[:,0].detach().cpu().numpy(), Y[:,1].detach().cpu().numpy()).sum()
                train_data.set_description('Train loss: {0:.4f}'.format(loss_val))
                loss.backward()
                classifier.clip_grad(self.D, 5)
                writer.add_scalar("Loss/train", loss_val, global_count)
#                 writer.add_scalar("Loss/val", loss_val, j)
                
                self.optimizer.step()
                del X,Y,f_X,loss
            ang_sep = np.rad2deg(sum_as/counts)
            writer.add_scalar("ang_sep", ang_sep, j)
            self.scheduler.step()
            if j%10==9:
                tqdm.write('After epoch {0} Train Ang Sep: {1:0.3f} '.format(j+1, ang_sep))

    def test(self):
        self.D.eval()
        counts = 0
        sum_as = 0.0
        for x, _, y in self.model_fetcher.test_data():
            counts += len(y)
            if cuda:
                X = Variable(torch.cuda.FloatTensor(x))
                Y = Variable(torch.cuda.FloatTensor(y))
            else:
                X = Variable(torch.FloatTensor(x))
                Y = Variable(torch.FloatTensor(y))
            f_X = self.D(X)
            sum_as += angular_separation(f_X[:, 0].detach().cpu().numpy(), f_X[:, 1].detach().cpu().numpy(),
                                         Y[:, 0].detach().cpu().numpy(), Y[:, 1].detach().cpu().numpy()).sum()
            del X, Y, f_X
        test_acc = sum_as/counts
        print('Final Test Accuracy: {0:0.3f}'.format(test_acc))
        return test_acc

if __name__ == "__main__":
    test_accs = []
    for i in range(num_repeats):
        print('='*30 + ' Start Run {0}/{1} '.format(i+1, num_repeats) + '='*30)
        t = PointCloudTrainer()
        t.train()
        acc = t.test()
        test_accs.append(acc)
        print('='*30 + ' Finish Run {0}/{1} '.format(i+1, num_repeats) + '='*30)
    print('\n')
    if num_repeats > 2:
        try:
            print('Test accuracy: {0:0.2f} '.format(np.mean(test_accs)) + unichr(177).encode('utf-8') + ' {0:0.3f} '.format(np.std(test_accs)))
        except:
            print('Test accuracy: {0:0.2f} +/-  {0:0.3f} '.format(np.mean(test_accs), np.std(test_accs)))
            
        writer.flush()
    writer.close()
    model_ouput_path = f'point_cloud_compton_{datetime.now()}.pkl'
    torch.save(t.D.state_dict(), model_ouput_path)
