import torch
import numpy as np
import time
import os
import argparse

from model import SegNet
from dataset import tuSimpleDataset
from torch.utils.data import DataLoader
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
        loss_f = 0
        
        for batch_idx, (imgs, labels) in enumerate(train_dataloader):
            

            input_tensor = torch.autograd.Variable(imgs).cuda()
            target_tensor = torch.autograd.Variable(labels).cuda()
            
            softmaxed_tensor = model(input_tensor)
            
            optimizer.zero_grad()
            loss = criterion(softmaxed_tensor, target_tensor)
            loss.backward()
            optimizer.step()
            
            loss_f += loss.float()
            prediction_f = softmaxed_tensor.float()
            
            if batch_idx % LOG_INTERVAL == 0:
                print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(imgs), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss.item()))

                #Tensorboard
                info = {'loss': loss.item()}

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, batch_idx + 1)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), batch_idx + 1)
                    logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), batch_idx + 1)

                # 3. Log training images (image summary)
                info = {'images': input_tensor.view(-1, 224, 224)[:10].cpu().numpy(), 'labels':target_tensor.view(-1, 224, 224)[:10].cpu().numpy()}

                for tag, images in info.items():
                    logger.image_summary(tag, images, batch_idx + 1)
            
        dt = time.time() - t_start
        is_better = loss_f < prev_loss
        
        if is_better:
            prev_loss = loss_f
            torch.save(model.state_dict(), "model_best.pth")
            
        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch+1, loss_f, dt))


if __name__ == "__main__":
   logger = Logger('./logs')
   
   #train_path = '/data/tuSimple/train_set/'
   train_path = args.train_path
   test_path = '/data/tuSimple/test_set/'
   #json_0313_path = '/data/tuSimple/train_set/label_data_0313.json'
   #json_0531_path = '/data/tuSimple/train_set/label_data_0531.json'
   #json_0601_path = '/data/tuSimple/train_set/label_data_0601.json'
   
   train_dataset = tuSimpleDataset(train_path, size=SIZE)
   test_dataset = tuSimpleDataset(test_path, size=SIZE)
   
   train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

   model = SegNet(input_ch=INPUT_CHANNELS, output_ch=OUTPUT_CHANNELS).cuda() 
   if os.path.isfile("model_best.pth"):
       model.load_state_dict(torch.load("model_best.pth"))
   criterion = torch.nn.CrossEntropyLoss().cuda()
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

   train()
