import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn
import Levenshtein as L
import csv
import pickle
import network
import videonetwork
import pdb

f = open('processed.pickle', 'rb')
data = pickle.load(f)

N = len(data)
train_data = data[:int(0.8*N)]
dev_data = data[int(0.8*N):]


class SpeechDataset(Dataset):
    def __init__(self,data, test=False):
        self.data = data
        self.test = test
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = torch.Tensor(data[i][j]).cuda()

    def __getitem__(self,i):
        return self.data[i] 

    def __len__(self):
        return len(self.data)


class LanguageModel(nn.Module):   
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.videonet = network.VideoNet()     
        #self.videonet = videonetwork.lipreading('temporalConv')
        #self.videonet.load_state_dict(torch.load('Video_only_model.pt'))
        self.magnet= network.MagnitudeSubNet()
        self.phasenet = network.PhaseSubNet()
        #self.phase_criterion = torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-8) → Tensor

    def forward(self, inputs):

        video = inputs[0]
        noisyMagnitude = inputs[1]
        cleanMagnitude = inputs[2]
        noisyPhase = inputs[3]
        cleanPhase = inputs[4]
        phasenet_output = torch.tensor([]).cuda()
        magnet_output = torch.tensor([]).cuda()

        for i in range(video.shape[1]):
            visual = video[:, i, 8:120, 8:120, :].unsqueeze(1) #batch * t * w * h * c
            #but i want batch*t*c *w*h
            visual = visual.contiguous().view(visual.shape[0], visual.shape[1], visual.shape[4], visual.shape[2], visual.shape[3])
            visual = self.videonet(visual)
            #clean_magn = torch.cat(((self.magnet(visual, noisyMagnitude[:,i]).unsqueeze(1)),magnet_output),dim = 1) #this is buggy, Danendra's job is to get this working
            clean_phase = self.phasenet(noisyPhase[:,i].squeeze(1), cleanMagnitude[:,i].squeeze(1)).unsqueeze(1)
            phasenet_output = torch.cat((phasenet_output, clean_phase), dim=1)

        magn_loss = F.l1_loss(cleanMagnitude, cleanMagnitude, reduce=False).sum((1, 2, 3)).mean(dim=0)
        phase_similarity = -1*F.cosine_similarity(phasenet_output, cleanPhase, dim=2).sum((1, 2)).mean(dim=0)
        
        return magn_loss, phase_similarity, magnet_output, phasenet_output

# model trainer
class Trainer:
    def __init__(self, model, train_loader, val_loader, max_epochs=1):        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader        
        self.epochs = 0
        self.max_epochs = max_epochs
        self.optimizer = torch.optim.Adam(model.parameters(),lr=0.0001, weight_decay=1e-6)
  


    def train_epoch_packed(self, epoch, resume=0):
        batch_id=0
        sum_loss = 0
        sum_similarity = 0
        freq = 1
        if(resume):
            checkpoint = torch.load('tmp/model'+ str(resume) +'.pkl')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])            

        for inputs in self.train_loader: # lists, presorted, preloaded on GPU  
            self.model.train() 
            #print(batch_id)
            batch_id += 1
            #print(epoch, batch_id)
            magn_loss, phase_similarity, magnet_output, phasenet_output = self.model(inputs)           
            sum_loss += magn_loss.item()
            sum_similarity += phase_similarity.item()
            var_magn_loss = Variable(magn_loss.data, requires_grad=True)
            self.optimizer.zero_grad()
            var_magn_loss.backward()
            phase_similarity.backward()
            self.optimizer.step()

            if batch_id % freq == 0:
                lpw = sum_loss / freq
                spw = sum_similarity/ freq
                sum_loss = 0
                #avg_valid_loss, avg_ldist = self.run_eval()
                avg_valid_loss = 0
                avg_ldist = 0
                print("Epoch:", epoch, " batch_id:", batch_id, " Avg train magn loss:", lpw, " Avg magnitude similarity:", spw)
                f = open('logger.txt', 'a')
                f.write(str(epoch) + ' ' + str(batch_id) + ' ' + str(lpw) + ' ' + str(spw) + '\n')
                f.close()                               


    def run_eval(self):
        self.model.eval()
        sum_loss = 0
        sum_similarity = 0
        batch_id=0
       
        for inputs in self.val_loader: # lists, presorted, preloaded on GPU  
            self.model.eval() 
            #print(batch_id)
            batch_id += 1
            #print(epoch, batch_id)
            magn_loss, phase_similarity, magnet_output, phasenet_output = self.model(inputs)           
            sum_loss += magn_loss.item()
            sum_similarity += phase_similarity.item()  
        return sum_loss/batch_id, sum_similarity / batch_id


    def run_test(self, epoch):        
        checkpoint = torch.load('tmp/model'+ str(epoch) +'.pkl')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()
        val_loss = 0
        batch_id = 0
        ls = 0.
        prediction = []
        '''
        for j in range(len(testX)):
            inputs = torch.FloatTensor([testX[j]]).cuda()
            i_lens = torch.IntTensor([len(inputs)]).cuda()
            print(j)'''
        for inputs, i_lens in self.test_loader:
            batch_id += 1
            print(batch_id)
            
            outputs = self.model(inputs, i_lens)            
            probs = F.softmax(outputs, dim=2)
            
            output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=i_lens)
            batch_ls= 0.0

            pred = ""

            for i in range(output.size(0)):
                pred = "".join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]]) 
                print(pred)

            prediction.append(pred) 

        N = len(prediction)
        print(N)

        

    def save(self, epoch):        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, 'tmp/model'+ str(epoch) +'.pkl')



model = LanguageModel().cuda()
train_dataset = SpeechDataset(train_data)
val_dataset = SpeechDataset(dev_data)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1)

trainer = Trainer(model, train_loader, val_loader, max_epochs = 10000)
resume = -1
flag = 1

for i in range(resume + 1, 100000):
    if(flag == 0):
        trainer.train_epoch_packed(i, resume=resume)
        flag = 1
    else:
        trainer.train_epoch_packed(i)

    #trainer.run_eval()
    avg_valid_magn_loss, avg_valid_phase_similarity = trainer.run_eval()

    print("Epoch:", i, " Avg_valid_magn_loss:", avg_valid_magn_loss, " avg_valid_phase_similarity:", avg_valid_phase_similarity)
    f = open('valid_logger.txt', 'a')
    f.write(str(i) + ' ' + ' ' + str(avg_valid_magn_loss) + ' ' + str(avg_valid_phase_similarity) + '\n')
    f.close()
    trainer.save(i)

trainer.run_test(76)
