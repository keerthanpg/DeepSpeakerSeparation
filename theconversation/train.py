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
			data[i]['video'] = torch.Tensor(data[i]['video']).cuda()
			data[i]['noisyMagnitude'] = torch.Tensor(data[i]['noisyMagnitude']).cuda()
			data[i]['noisyPhaseImag'] = torch.Tensor(data[i]['noisyPhaseImag']).cuda()
			data[i]['noisyPhaseReal'] = torch.Tensor(data[i]['noisyPhaseReal']).cuda()		
			data[i]['cleanMagnitude'] = torch.Tensor(data[i]['cleanMagnitude']).cuda()			
			data[i]['cleanPhaseImag'] = torch.Tensor(data[i]['cleanPhaseImag']).cuda()
			data[i]['cleanPhaseReal'] = torch.Tensor(data[i]['cleanPhaseReal']).cuda()			
		
	def __getitem__(self,i):
		return self.data[i]	

	def __len__(self):
		return len(self.data)


class LanguageModel(nn.Module):    
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.magnet= network.MagnitudeSubNet()
        self.phasenet = network.PhaseSubNet()

    def forward(self, inputs):
    	print(inputs.shape)
        return scores


# model trainer
class Trainer:
    def __init__(self, model, train_loader, val_loader, max_epochs=1):        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader        
        self.epochs = 0
        self.max_epochs = max_epochs
        self.optimizer = torch.optim.Adam(model.parameters(),lr=0.0001, weight_decay=1e-6)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()     


    def train_epoch_packed(self, epoch, resume=0):
        batch_id=0
        sum_loss = 0
        freq = 10
        if(resume):
            checkpoint = torch.load('tmp/model'+ str(resume) +'.pkl')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])            

        for inputs in self.train_loader: # lists, presorted, preloaded on GPU  
            self.model.train()          
            #print(batch_id)
            batch_id += 1
            #print(epoch, batch_id)
            outputs = self.model(inputs)
            outputs = torch.transpose(outputs, 0, 1) #sequence length * batchsize *outputdim            
            loss = self.criterion(outputs.cpu(), torch.cat(targets).cpu(), i_lens.cpu(), t_lens.cpu()).cuda()
            sum_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_id % freq == 0:
                lpw = sum_loss / freq
                sum_loss = 0
                #avg_valid_loss, avg_ldist = self.run_eval()
                avg_valid_loss = 0
                avg_ldist = 0
                print("Epoch:", epoch, " batch_id:", batch_id, " Avg train loss:", lpw," Avg train perplexity:",np.exp(lpw))
                f = open('logger.txt', 'a')
                f.write(str(epoch) + ' ' + str(batch_id) + ' ' + str(lpw) + '\n')
                f.close()                               


    def run_eval(self):
        self.model.eval()
        val_loss = 0
        batch_id=0
        ls = 0.
        n_words = 0
        for inputs, targets, i_lens, t_lens in self.val_loader:
            n_words += sum(i_lens)
            batch_id += 1
            outputs = self.model(inputs, i_lens)
            outputs = torch.transpose(outputs, 0, 1) #sequence length * batchsize *outputdim            
            loss = self.criterion(outputs.cpu(), torch.cat(targets).cpu(), i_lens.cpu(), t_lens.cpu())
            val_loss += loss.item()
            outputs = torch.transpose(outputs, 0, 1) #batch * sequence length *outputdim
            probs = F.softmax(outputs, dim=2)
            #print(i_lens.shape)

            output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=i_lens)
            batch_ls= 0.0

            for i in range(output.size(0)):
                pred = "".join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])
                true = "".join(self.label_map[l] for l in targets[i])
                #print("Pred: {}, True: {}".format(pred, true))
                batch_ls += L.distance(pred, true)
            #assert pos == labels.size(0)
            ls += batch_ls/len(inputs)
        return val_loss/batch_id, ls / batch_id


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
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=32)

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
    avg_valid_loss, avg_ldist = trainer.run_eval()
    print("Epoch:", i, " Avg_valid_loss:", avg_valid_loss, " avg_levenstein:", avg_ldist)
    f = open('valid_logger.txt', 'a')
    f.write(str(i) + ' ' + ' ' + str(avg_valid_loss) + ' ' + str(avg_ldist) + '\n')
    f.close()
    trainer.save(i)

trainer.run_test(76)
