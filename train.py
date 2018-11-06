import torch
import numpy as np
import time
import os
import argparse

from segnet import SegNet
from enet import ENet
from loss import DiscriminativeLoss
from dataset import tuSimpleDataset
from logger import Logger

parser = argparse.ArgumentParser(description="Train model")

parser.add_argument('--train-path', required=True)
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batch-size', type=int, default=10, help='batch size')
parser.add_argument('--img-size', type=int, nargs='+', default=[224, 224], help='image resolution: [width height]')
parser.add_argument('--epoch', type=int, default=100)

args = parser.parse_args()

INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 2
LEARNING_RATE = args.lr #1e-5
BATCH_SIZE = args.batch_size #20
NUM_EPOCHS = args.epoch #100
LOG_INTERVAL = 10
SIZE = [args.img_size[0], args.img_size[1]] #[224, 224]

def train():
    # refer from : https://github.com/Sayan98/pytorch-segnet/blob/master/src/train.py
    is_better = True
    prev_loss = float('inf')
    
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        t_start = time.time()
        loss_f = []

        for batch_idx, (imgs, sem_labels, ins_labels) in enumerate(train_dataloader):
            loss = 0

            img_tensor = torch.autograd.Variable(imgs).cuda()
            sem_tensor = torch.autograd.Variable(sem_labels).cuda()
            ins_tensor = torch.autograd.Variable(ins_labels).cuda()

            # Init gradients
            optimizer.zero_grad()

            # Predictions
            sem_pred, ins_pred = model(img_tensor)

            # Discriminative Loss
            disc_loss = criterion_disc(ins_pred, ins_tensor, [5] * len(img_tensor))
            loss += disc_loss

            # CrossEntropy Loss

            ce_loss = criterion_ce(sem_pred.permute(0,2,3,1).contiguous().view(-1,OUTPUT_CHANNELS),
                                   sem_tensor.view(-1))
            loss += ce_loss

            loss.backward()
            optimizer.step()

            loss_f.append(loss.cpu().data.numpy())

            if batch_idx % LOG_INTERVAL == 0:
                print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(imgs), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss.item()))

                #Tensorboard
                info = {'loss': loss.item(), 'ce_loss': ce_loss.item(), 'disc_loss': disc_loss.item(), 'epoch': epoch}

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, batch_idx + 1)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), batch_idx + 1)
                    # logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), batch_idx + 1)

                # 3. Log training images (image summary)
                info = {'images': img_tensor.view(-1, 3, SIZE[0], SIZE[1])[:BATCH_SIZE].cpu().numpy(),
                        'labels': sem_tensor.view(-1, SIZE[0], SIZE[1])[:BATCH_SIZE].cpu().numpy(),
                        'sem_preds': sem_pred.view(-1, 2, SIZE[0], SIZE[1])[:BATCH_SIZE,1].data.cpu().numpy(),
                        'ins_preds': ins_pred.view(-1, SIZE[0], SIZE[1])[:BATCH_SIZE*5].data.cpu().numpy()}

                for tag, images in info.items():
                    logger.image_summary(tag, images, batch_idx + 1)
            
        dt = time.time() - t_start
        is_better = np.mean(loss_f) < prev_loss
        scheduler.step()
        
        if is_better:
            prev_loss = np.mean(loss_f)
            print("\t\tBest Model.")
            torch.save(model.state_dict(), "model_best.pth")
            
        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s, Lr: {:2f}".format(epoch+1, np.mean(loss_f), dt, optimizer.param_groups[0]['lr']))


if __name__ == "__main__":
   logger = Logger('./logs')
   
   train_path = args.train_path
   train_dataset = tuSimpleDataset(train_path, size=SIZE)
   train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

   #model = SegNet(input_ch=INPUT_CHANNELS, output_ch=OUTPUT_CHANNELS).cuda() 
   model = ENet(input_ch=INPUT_CHANNELS, output_ch=OUTPUT_CHANNELS).cuda() 
   if os.path.isfile("model_best.pth"):
       print("Loaded model_best.pth")
       model.load_state_dict(torch.load("model_best.pth"))

   criterion_ce = torch.nn.CrossEntropyLoss().cuda()
   criterion_disc = DiscriminativeLoss(delta_var=0.1,
                                       delta_dist=0.6,
                                       norm=2,
                                       usegpu=True).cuda()
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
   scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30,40,50,60,70,80], gamma=0.9)

   train()
