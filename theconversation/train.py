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

f = open('extracted_new.pickle', 'rb')
data = pickle.load(f)


class SpeechDataset(Dataset):
	def __init__(self,utterances, transcription=None, test=False):
		self.utterances= [torch.FloatTensor(l) for l in utterances]
		self.test = test
		if(not self.test):
			charmap = []
			for sentence in transcription:
				s_trans = [vocab_lookup['<sos>']]
				N = len(sentence)
				for i in range(N):
					for char in sentence[i]:
						if chr(char) in vocab_lookup:
							s_trans.append(vocab_lookup[chr(char)])    
						else:
							s_trans.append(vocab_lookup['<unk>'])
					if(i != N-1):
						s_trans.append(vocab_lookup[' '])
				s_trans.append(vocab_lookup['<eos>']) 
				charmap.append(s_trans)         
			self.transcription=[torch.LongTensor(l) for l in charmap]
		
	def __getitem__(self,i):
		utterance = self.utterances[i]
		if(not self.test):
			transcription = self.transcription[i]      
			return utterance.cuda(), transcription.cuda()
		else:   
			return utterance.cuda(), 1

	def __len__(self):
		return len(self.utterances)

class CustomDataSet(Dataset):

    def __init__ (self, mode):
        self.mode = mode

    def __getitem__(self,index):
        if self.mode is 'train':
            trainVideo, trainAudioMagnitude, trainAudioPhase = data_loader.getData(self.mode)
    def __len__(self):
        return self.trainVideo.shape[0]