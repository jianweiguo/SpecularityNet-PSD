from os.path import join, basename
from options.specularitynet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
from data.transforms import __scale_width
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util

from data import spec
from torch.utils.data import DataLoader

opt = TrainOptions().parse()

opt.isTrain = False
cudnn.benchmark = True
opt.no_log =True
opt.display_id=0
opt.verbose = False

dataset_wild = spec.TestDataset('/path/',imgsize='small')
dataloader_wild = DataLoader(dataset_wild,1,num_workers=opt.nThreads,shuffle=not opt.serial_batches,drop_last=False)

engine = Engine(opt)

engine.test(dataloader_wild, savedir=join('./results','test'))


