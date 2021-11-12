from os.path import join
from options.specularitynet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import data
from data import spec
from torch.utils.data import DataLoader

opt = TrainOptions().parse()

cudnn.benchmark = True

opt.display_freq = 10

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

#dataset_filtered &dataset_appended: random polarization angles
dataset_filtered = spec.SpecDataset(opt,'/PSD_Dataset/filtered',imgsize='small')
dataloader_filtered = DataLoader(dataset_filtered,opt.batchSize,num_workers=opt.nThreads,shuffle=not opt.serial_batches,drop_last=False)
#dataset_appended = spec.SpecDataset(opt,'/PSD_Dataset/appended',imgsize='small')
#dataloader_appended = DataLoader(dataset_appended,opt.batchSize,num_workers=opt.nThreads,shuffle=not opt.serial_batches,drop_last=False)
#dataset_aligned: fixed polarization angles
dataset_aligned = spec.GroupDataset(opt,'/PSD_Dataset/aligned',imgsize='small',groups=800,idxs=12,idxd=1,idxis=[7],name="group-{:04d}-idx-{:02d}.png",freq=opt.freq,any_valid=False)
dataloader_aligned = DataLoader(dataset_aligned,opt.batchSize,num_workers=opt.nThreads,shuffle=not opt.serial_batches,drop_last=False)


dataset_val = spec.GroupDataset(opt,'/PSD_Dataset/val',imgsize='small',groups=range(1001,1100),idxs=12,idxd=1,idxis=[7],name="group-{:04d}-idx-{:02d}.png",freq=opt.freq,any_valid=False)
dataloader_val = DataLoader(dataset_val,opt.batchSize,num_workers=opt.nThreads,shuffle=not opt.serial_batches,drop_last=False)


dataset_wild = spec.TestDataset('/PSD_Dataset/wild',imgsize='small')
dataloader_wild = DataLoader(dataset_wild,1,num_workers=opt.nThreads,shuffle=not opt.serial_batches,drop_last=False)

"""Main Loop"""
engine = Engine(opt)

def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)

engine.model.opt.lambda_gan = 0
lr = 1e-4

while engine.epoch < 80:
    if engine.epoch >= 20:
        engine.model.opt.lambda_gan = 0.01 # gan loss is added after epoch 10
    if (engine.epoch+1)%5 == 0:
        lr_now = max(1e-5,lr*0.8**((engine.epoch+1)/5))
        set_learning_rate(lr_now)
    if True:
        print("coast training ...")
        engine.train(dataloader_aligned)
        engine.train(dataloader_filtered)
        engine.train(dataloader_val)
        engine.train(dataloader_appended)
        # engine.train(dataloader_train)
        engine.epoch += 1
        if engine.epoch % 5 == 0:
            # engine.eval(dataloader_aligned, dataset_name='dataset_aligned', savedir=join('./results','aligned'))
            # engine.eval(dataloader_unaligned, dataset_name='dataset_unaligned', savedir=join('./results','unaligned'))
            # engine.eval(dataloader_val, dataset_name='dataset_val', savedir=join('./results','val'))
            # engine.eval(dataloader_test, dataset_name='dataset_test', savedir=join('./results','test'))
            engine.test(dataloader_wild, savedir=join('./results','wild'))
