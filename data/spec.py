import torch
import os.path
from os.path import join
import numpy as np
import cv2
import math
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class SpecDataset(torch.utils.data.Dataset):
    def __init__(self, opt, datadir, dirA='spec', dirB='nospec',imgsize=None):
        super(SpecDataset, self).__init__()
        self.opt = opt
        self.datadir = datadir
        self.dirA = dirA
        self.dirB = dirB
        self.fnsA = sorted(os.listdir(join(datadir,dirA)))
        #self.fnsB = sorted(os.listdir(join(datadir,dirB)))
        self.fnsB = self.fnsA
        self.imgsize = imgsize
        # np.random.seed(0)
        print('Load {} items in {} ...'.format(len(self.fnsA),datadir))

    def __getitem__(self, index):
        fnA = self.fnsA[index]
        fnB = self.fnsB[index]
        t_img = cv2.imread(join(self.datadir, self.dirB, fnB))
        m_img = cv2.imread(join(self.datadir, self.dirA, fnA))
        # print(self.imgsize)
        if np.random.rand() < self.opt.fliplr:
            t_img = cv2.flip(t_img,1)
            m_img = cv2.flip(m_img,1)
        if np.random.rand() < self.opt.flipud:
            t_img = cv2.flip(t_img,0)
            m_img = cv2.flip(m_img,0)
        if self.imgsize == 'middle':
            size = (768,512)
        elif self.imgsize == 'small':
            size = (384,256)
        else:
            # size = (m_img.shape[1],m_img.shape[0])
            if m_img.shape[0] < m_img.shape[1]:
                size = (int(256*m_img.shape[1]/m_img.shape[0]),256)
            else:
                size = (256,int(256*m_img.shape[0]/m_img.shape[1]))
        if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]) and not self.imgsize is None:
            scale = int(math.log2(min(m_img.shape[0]/size[1],m_img.shape[1]/size[0])))
            for i in range(0,scale):
                m_img = cv2.pyrDown(m_img)
                t_img = cv2.pyrDown(t_img)
            if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]) or not (t_img.shape[0] == size[1] and t_img.shape[1] == size[0]):
                m_img = cv2.resize(m_img,size,cv2.INTER_AREA)
                t_img = cv2.resize(t_img,size,cv2.INTER_AREA)

        t_img = cv2.cvtColor(t_img,cv2.COLOR_BGR2RGB)
        m_img = cv2.cvtColor(m_img,cv2.COLOR_BGR2RGB)

        M = np.transpose(np.float32(m_img)/255.0,(2,0,1))
        T = np.transpose(np.float32(t_img)/255.0,(2,0,1))
        delta = M-T
        mask = 0.3*delta[0]+0.59*delta[1]+0.11*delta[2]
        mask = np.float32(mask>0.707*mask.max())
        if self.opt.noise:
            M = M+np.random.normal(0,2/255.0,M.shape).astype(np.float32)
            #T = T+np.random.normal(0,1/255.0,T.shape).astype(np.float32)
        data = {'input': M,  'target_t': T, 'fn': fnA[:-4], 'mask':mask}
        return data

    def __len__(self):
        return len(self.fnsA)


class GroupDataset(torch.utils.data.Dataset):
    def __init__(self, opt,datadir,imgsize=None,groups=95,idxs=12,idxd=1,idxis=[7],name="group-{:04d}-idx-{:02d}.png",freq=-1,any_valid=True):
        super(GroupDataset, self).__init__()
        self.opt = opt
        self.datadir = datadir
        self.imgsize = imgsize
        self.freq = freq
        
        if isinstance(groups,int):
            self.groups = list(range(1,groups+1))
        else:
            self.groups = list(groups)
        self.idxd = idxd
        if isinstance(idxs,int):
            assert (not idxd in idxis) and idxd <= idxs
            self.idxs = []
            for idx in range(1,idxs+1):
                if (not idx == idxd) and (not idx in idxis):
                    self.idxs.append(idx)
        else:
            self.idxs = list(idxs)
        self.name = name
        self.build(groups,any_valid)
        print('Load {} items in {} ...'.format(len(self.pairs),datadir))
        if self.freq > 0 and self.freq < 1:
            print('Select {} items ...'.format(int(len(self.pairs)*self.freq)))

    def build(self,groups,any_valid):
        self.pairs = []
        if any_valid:
            for g in self.groups:
                if os.path.exists(os.path.join(self.datadir,self.name.format(g,self.idxd))):
                    for idx in self.idxs:
                        if os.path.exists(os.path.join(self.datadir,self.name.format(g,idx))):
                            self.pairs.append({'input':self.name.format(g,idx),'target':self.name.format(g,self.idxd)})
        else:
            for g in self.groups:
                if not os.path.exists(os.path.join(self.datadir,self.name.format(g,self.idxd))):
                    continue
                group_pairs = []
                valid = True
                for idx in self.idxs:
                    if os.path.exists(os.path.join(self.datadir,self.name.format(g,idx))):
                        group_pairs.append({'input':self.name.format(g,idx),'target':self.name.format(g,self.idxd)})
                    else:
                        valid = False
                        break
                if valid:
                    self.pairs += group_pairs

    def __getitem__(self, index):
        if self.freq > 0 and self.freq < 1:
            index = np.random.randint(len(self.pairs))
        fnA = self.pairs[index]['input']
        fnB = self.pairs[index]['target']
        m_img = cv2.imread(join(self.datadir, fnA))
        t_img = cv2.imread(join(self.datadir, fnB))
        # print(self.imgsize)
        if np.random.rand() < self.opt.fliplr:
            t_img = cv2.flip(t_img,1)
            m_img = cv2.flip(m_img,1)
        if np.random.rand() < self.opt.flipud:
            t_img = cv2.flip(t_img,0)
            m_img = cv2.flip(m_img,0)
        if self.imgsize == 'middle':
            size = (768,512)
        elif self.imgsize == 'small':
            size = (384,256)
        else:
            # size = (m_img.shape[1],m_img.shape[0])
            if m_img.shape[0] < m_img.shape[1]:
                size = (int(256*m_img.shape[1]/m_img.shape[0]),256)
            else:
                size = (256,int(256*m_img.shape[0]/m_img.shape[1]))
        if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]):
            scale = int(math.log2(min(m_img.shape[0]/size[1],m_img.shape[1]/size[0])))
            for i in range(0,scale):
                m_img = cv2.pyrDown(m_img)
                t_img = cv2.pyrDown(t_img)
            if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]) or not (t_img.shape[0] == size[1] and t_img.shape[1] == size[0]):
                m_img = cv2.resize(m_img,size,cv2.INTER_AREA)
                t_img = cv2.resize(t_img,size,cv2.INTER_AREA)
        t_img = cv2.cvtColor(t_img,cv2.COLOR_BGR2RGB)
        m_img = cv2.cvtColor(m_img,cv2.COLOR_BGR2RGB)
 
        M = np.transpose(np.float32(m_img)/255.0,(2,0,1))
        T = np.transpose(np.float32(t_img)/255.0,(2,0,1))
        delta = M-T
        mask = 0.3*delta[0]+0.59*delta[1]+0.11*delta[2]
        mask = np.float32(mask>0.507*mask.max())
        if self.opt.noise:
            M = M+np.random.normal(0,2/255.0,M.shape).astype(np.float32)
            #T = T+np.random.normal(0,1/255.0,T.shape).astype(np.float32)
        data = {'input': M,  'target_t': T, 'fn': fnA[:-4], 'mask':mask}
        return data

    def __len__(self):
        if self.freq > 0 and self.freq < 1:
            return int(len(self.pairs)*self.freq)
        else:
            return len(self.pairs)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, datadir,imgsize='small'):
        super(TestDataset, self).__init__()
        self.datadir = datadir
        self.fns = sorted(os.listdir(datadir))
        self.imgsize = imgsize
        print('Load {} items in {} ...'.format(len(self.fns),datadir))
        
    def __getitem__(self, index):
        fn = self.fns[index]
        m_img = cv2.imread(join(self.datadir, fn))

        # print(self.imgsize)
        assert self.imgsize in ['middle','small','origin']
        if self.imgsize == 'middle':
            if m_img.shape[0] < m_img.shape[1]:
                size = (int(512*m_img.shape[1]/m_img.shape[0]),512)
            else:
                size = (512,int(512*m_img.shape[0]/m_img.shape[1]))
        else:
            if m_img.shape[0] < m_img.shape[1]:
                size = (int(256*m_img.shape[1]/m_img.shape[0]),256)
            else:
                size = (256,int(256*m_img.shape[0]/m_img.shape[1]))
        if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]):
            scale = int(math.log2(min(m_img.shape[0]/size[1],m_img.shape[1]/size[0])))
            for i in range(0,scale):
                m_img = cv2.pyrDown(m_img)
            if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]):
                m_img = cv2.resize(m_img,size,cv2.INTER_AREA)

        m_img = cv2.cvtColor(m_img,cv2.COLOR_BGR2RGB)
        M = np.transpose(np.float32(m_img)/255.0,(2,0,1))
        data = {'input': M, 'target_t': torch.zeros([1,0]), 'fn': fn[:-4], 'mask' : torch.zeros([1,0])}
        return data

    def __len__(self):
        return len(self.fns)
