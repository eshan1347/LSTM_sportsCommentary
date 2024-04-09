import torch
import pandas as pd
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence
from pathlib import Path
import argparse
# import tensorflow as tf

data = pd.read_csv("events.csv")

device='cuda:0' if torch.cuda.is_available() else 'cpu'
batch_size = 16
lr = 1e-3
bidi = False
n_layers = 2
n_emb = 50
dropout = 0.2
bias = True
eval_iters = 500
iters = 5000
clip_value = 1.0

#Mappings : time | event_type | side | is_goal | assist_method | fast_break
eventM = {
    0: 'Announcement',
    1: 'Attempt',
    2: 'Corner',
    3: 'Foul',
    4: 'Yellow Card',
    5: 'Second Yellow Card',
    6: 'Red Card',
    7: 'Substitution',
    8: 'Free kick won',
    9: 'Offside',
    10: 'Hand ball',
    11: 'Penalty conceded'
}

sideM = {
    1: 'Home',
    2: 'Away'
}

goalM = {
    1 : 'Goal!',
    0: 'Not a Goal'
}

assistM = {
    0	: 'None',
    1	: 'Pass',
    2	: 'Cross',
    3	: 'Headed pass',
    4	: 'Through ball'
}

fastM = {
    1 : 'fast break',
    0 : 'not applicable'
}

data1 = data.drop(columns=['id_odsp','id_event','sort_order'])
data2  = data1.drop(columns=['player2','player_in','player_out','event_type2','shot_place','shot_outcome','bodypart','situation','location'])
data2 = data2.iloc[:20000,:]
data2['event_type'] = data2['event_type'].replace(eventM)
data2['side'] = data2['side'].replace(sideM)
data2['is_goal'] = data2['is_goal'].replace(goalM)
data2['assist_method'] = data2['assist_method'].replace(assistM)
data2['fast_break'] = data2['fast_break'].replace(fastM)
# data2['time'] = data2['time'].astype(str)
data2.fillna('None' , inplace=True)
for i in range(len(data2)):
  data2.iloc[i,0] = str(data2.iloc[i,0])
dataX = data2.drop(columns=['text'])
dataY = data2['text']
for i,y in enumerate(dataY):
  dataY[i] = '<Start> ' + y + ' <End>'

def getMax(Y):
  max = 0
  for i in Y:
    if max < len(i):
      max = len(i)
  return max

def padSeq(L, l, c):
  L1 = L[:]
  diff = l - len(L)
  if diff != 0:
    for i in range(diff):
      L1.append(c)
  return L1

y_trainT = []
for i in dataY.values:
  for j in i.split():
    y_trainT.append(j)

X_T = []
for i in dataX.values:
  for j in i:
    X_T.append(j)

d_T = X_T + y_trainT
d_T.append(' ')
d_T.append('/n')
d_T = sorted(set(d_T))
vocab_size = len(d_T)
hiddenDim = vocab_size
# wordVec = Word2Vec(d_T,vector_size=100,window=5,min_count=1)

str2int = { ch:i for i,ch in enumerate(d_T)}
int2str = { i:ch for i,ch in enumerate(d_T)}
enc = lambda x : [str2int[i] for i in x]
dec = lambda x : [int2str[i] for i in x]

encDataX = [enc(i) for i in dataX.values]
encDataY = [enc(i.split()) for i in dataY.values]

X_train, X_test, y_train, y_test = train_test_split(encDataX, encDataY, test_size=0.2, random_state=42)



class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.embedT = nn.Embedding(vocab_size,n_emb)
    self.rnn = nn.GRU(n_emb,hiddenDim,n_layers,bias=bias,dropout=dropout,bidirectional=bidi)

  def forward(self,x):
    x = self.embedT(x)
    B,T,C = x.shape
    h0 = self.init_state(T,x)
    logits, h0 = self.rnn(x,h0)
    return logits, h0

  def init_state(self,T,x):
    return torch.zeros(n_layers,T,hiddenDim, device=device, dtype=x.dtype)

class Decoder(nn.Module):
  def __init__(self, encoder):
    super().__init__()
    self.enc = encoder
    self.embedT = nn.Embedding(vocab_size,n_emb)
    self.dec = nn.GRU(n_emb,hiddenDim,n_layers,bias=bias, dropout=dropout,bidirectional=bidi)
    self.ff = nn.Sequential(nn.Linear(hiddenDim, n_layers*hiddenDim),nn.ReLU(),nn.Linear(n_layers*hiddenDim,hiddenDim),nn.Dropout(dropout))
    self.softmax = nn.Softmax(dim=-1)
    self.apply(self._init_weights)

  def _init_weights(self,module):
    if isinstance(module, torch.nn.Linear):
      torch.nn.init.normal_(module.weight,mean = 0.0 , std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self,x,y=None):
    op, hn = self.enc(x)
    y1 = self.embedT(x) if y is None else self.embedT(y)
    ph0 = torch.zeros(hn.shape[0],y1.shape[1] , hn.shape[2], device=device, dtype=y1.dtype)
    ph0[:, :hn.shape[1], :] = hn
    logits, hn = self.dec(y1,ph0)
    logits = self.ff(logits)
    # print(f'y1 dt: {y1.dtype} | ph0 dt: {ph0.dtype} | h dt:{hn.dtype}')
    # logits, hn = torch.utils.checkpoint.checkpoint(self.dec, y1, ph0)
    # logits = self.softmax(logits)
    if y is None:
      loss = None
    else:
      B,T,H = logits.shape
      # print(f'Logits : {logits.shape} | op : {y.shape}')
      logits = logits.view(B*T,H)
      # print(f'Logits : {logits.shape} | op : {y.shape}')
      op = y.view(B*T)
      loss = nn.functional.cross_entropy(logits, op)
    return logits, loss

  def genTokens(self, iters, ip):
    for _ in range(iters):
      # ip_crop = ip[:,-]
      logits, loss = self.forward(ip)
      logits = logits[:,-1,:]
      probs = torch.nn.functional.softmax(logits,dim=-1)
      nxt = torch.multinomial(logits, num_samples=1)
      ip = torch.cat((ip,nxt),dim=1)
    return ip

encoder = Encoder().to(device)
model = Decoder(encoder).to(device)
# model = model.half()
# lencoder = Encoder()
# lmodel = Decoder(lencoder)
model.load_state_dict(torch.load(f='seq2seq.pt', map_location=torch.device(device)))


# context = torch.ones((1,1), dtype=torch.long, device=device)
res = lmodel.genTokens(9,c)
str = ''
for i in dec(res[0].tolist()):
  str += i + ' '

while True:
  n = input("Number of words: \n")
  prompt = input("Prompt['time', 'event_type', 'side', 'event_team', 'opponent', 'player','is_goal', 'assist_method', 'fast_break']:\n")
  context = torch.tensor(enc(prompt), dtype=torch.long, device=device)
  res = lmodel.genTokens(n,context)
  str = ''
  for i in dec(res[0].tolist()):
    str += i + ' '
  print(f'Completion! \n {str}')