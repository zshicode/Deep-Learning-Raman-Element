import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,accuracy_score
import os
import shutil
import seaborn
seaborn.set(style='whitegrid',font_scale=1.5)
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from collections import Counter

elem = ['Mg','Al','P','S','Fe','Cu']

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of representations')
parser.add_argument('--c', type=int, default=2,
                    help='Num of classes')
parser.add_argument('--d', type=int, default=1024,
                    help='Num of spectra dimension')
parser.add_argument('--model', type=str, default='NN',
                    help='Model')
parser.add_argument('--wavelength', type=str, default='780',
                    help='Excitation wavelength of data')
parser.add_argument('--raw', type=str, default='Processed',
                    help='Processed/RAW')                     

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

set_seed(args.seed,args.cuda)

def preprocess(raw,wavelength):
    for f in os.listdir('./RRUFF_data'):
        os.remove('./RRUFF_data/'+f)

    for f in os.listdir('./RRUFF_Raman_excellent_unoriented'):
        if raw in f:
            if wavelength+'_' in f or wavelength == '0':
                shutil.copy('./RRUFF_Raman_excellent_unoriented/'+f,'./RRUFF_data/'+f)

    name_cnt = []
    for f in os.listdir('./RRUFF_data'):
        fs = f.split('__')
        name_cnt.append(fs[0])

    counter = Counter(name_cnt)
    for f in os.listdir('./RRUFF_data'):
        fs = f.split('__')
        if counter[fs[0]] < 2: os.remove('./RRUFF_data/'+f)

def read_data(dataset_dir):
    file_list = os.listdir(dataset_dir)
    n_samples = len(file_list)
    a = np.zeros((n_samples,1024))
    i = 0
    name = []
    rid = []
    comp = []
    for file_name in file_list:
        fs = file_name.split('__')
        name.append(fs[0])
        rid.append(fs[1])
        mat,compound = read_spectrum_intensity(os.path.join(dataset_dir, file_name))
        comp.append(compound)
        raman_shift = np.linspace(100,1400,1024)
        intensity = np.interp(raman_shift,mat[:,0],mat[:,1],0,0)
        intensity -= min(intensity)
        #intensity = np.convolve(intensity,np.ones(3)/3,mode='same')
        intensity /= max(intensity)
        a[i] = intensity
        i += 1
    
    return np.around(a,4),name,rid,comp

def read_spectrum_intensity(file_path):
    if os.path.isfile(file_path):
        intensity = []
        with open(file_path, "r", encoding='utf-8') as f:
            data_raw = f.readlines()
            comp = ' '
            for line in data_raw:
                if line[0] != '#' and line[0] != '\n':
                    intensity.append(eval(line))
                else:
                    temp = line.split('=')
                    if temp[0] == '##IDEAL CHEMISTRY': comp = temp[1]
        
        return np.array(intensity), comp
    else:
        raise Exception("不存在该文件： ", file_path)

def find_ele(ele,comp):
    e = comp.find(ele)
    if e == -1: return 0
    if len(ele) > 1: return 1
    else:
        if e == len(comp)-1: return 1
        elif comp[e+1].islower(): return 0
        else: return 1

def foldcv(x,y,name):
    predict = np.zeros(len(y))
    pred = np.zeros(len(y))
    kf = KFold(n_splits=5,shuffle=True)
    for i,(train,test) in enumerate(kf.split(x,y)):
        xt = x[train]
        yt = y[train]
        xv = x[test]
        yv = y[test]
        pred[test],predict[test] = nnclassifier(xt,yt,xv,yv)

    return evaluate(y,pred,predict,name)

def evaluate(yv,pred,predict,name):
    confusion = confusion_matrix(yv, pred)
    fpr, tpr, th = roc_curve(yv, predict)
    rec = []
    for i in range(len(pred)):
        if yv[i]!=pred[i]: rec.append(i)
    
    acc = accuracy_score(yv,pred)
    auc = roc_auc_score(yv,predict)
    thres = th[np.argmax(tpr-fpr)]
    plt.figure()
    plt.title('%s, accuracy=%.3f' % (name,acc))
    seaborn.heatmap(confusion,annot=True,cbar=False,
        xticklabels=['Pred=0','Pred=1'],
        yticklabels=['True=0','True=1'],fmt='d')
    plt.ylim(2,0)
    plt.savefig('%s-conf.png' % name)
    plt.show()
    print('%s,#PositiveSample=%d,auc=%.3f' % (name,yv.sum(),auc))

class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()
        self.lin1 = nn.Linear(args.d,args.hidden)
        self.q = nn.Linear(args.hidden,args.hidden)
        self.k = nn.Linear(args.hidden,args.hidden)
        self.v = nn.Linear(args.hidden,args.hidden)
        self.lin2 = nn.Linear(args.hidden,1)

    def forward(self,x):
        x = self.lin1(x)
        w = torch.mm(self.q(x),self.k(x).t())/np.sqrt(args.hidden)
        x = x + torch.mm(torch.softmax(w,dim=-1),self.v(x))
        x = self.lin2(x)
        return x.squeeze()

def nnclassifier(xt,yt,xv,yv):
    xt = torch.from_numpy(xt).float()
    xv = torch.from_numpy(xv).float()
    yt = torch.Tensor(yt)
    clf = Attention()
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd)
    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()
    
    for e in range(args.epochs):
        clf.train()
        z = clf(xt)
        loss = F.mse_loss(z,yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # if e%10 == 0 and e!=0:
        #     print('Epoch %d | Lossp: %.4f' % (e, loss.item()))

    clf.eval()
    z = clf(xv)
    if args.cuda: z = z.cpu()
    predict = z.detach().numpy()
    predict -= min(predict)
    predict /= max(predict)
    fpr,tpr,th = roc_curve(yv,predict)
    pred = np.ones(len(predict))
    for i in range(len(predict)):
        if predict[i]<th[np.argmax(tpr-fpr)]: pred[i] = 0.0
    
    return pred,predict

# preprocess(args.raw,args.wavelength)
# x,name,rid,comp = read_data('./RRUFF_data')
# df = pd.DataFrame(x)
# df['name'] = name
# df['rid'] = rid
# df['comp'] = comp
# df.to_csv('780-process.csv',index=False)
df = pd.read_csv('780-process.csv')
name = df['name']
rid = df['rid']
comp = df['comp']
x = df.values[:,:1024].astype('float')

def test(ele):
    y = []
    for i in range(len(comp)):
        y.append(find_ele(ele,comp[i]))
    
    y = np.array(y)
    foldcv(x,y,ele)

for e in elem:
    test(e)