#!/usr/bin/python
'''
Extra comments by Sharan
Contains the utilities used for
loading, initating and running up the networks
for all training, validation and evaluation.
'''
import torch
import torch.nn as nn
from torch.nn import init # weight initializer
import functools
from torch.optim import lr_scheduler # for adjusting the learning rate
import numpy as np
import os
from Generators import *
from params import *


def write_log(log_values, model_name, log_dir="", log_type='loss', type_write='a'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_dir+"/"+model_name+"_"+log_type+".txt", type_write) as f:
        f.write(','.join(log_values)+"\n")

def get_model_funct(model_name): # gets the model function which will ultimately initialize the network (written below)
    if model_name == "G":
        return define_G

def define_G(opt, gpu_ids): 
    # This function takes the type of network we want (Encoder or Decoder), returns an initialized pytorch sequential container (the network basically)
    # combination of all helper functions written below
    # (includes creating the encoder/decoder/resnet block, normalization type, learning rate annealing, weight initialization, setting the compute)
    net = None
    input_nc    = opt.input_nc # input channels
    output_nc   = opt.output_nc # output channels
    ngf         = opt.ngf # depth of the first set of encoders
    net_type    = opt.base_model # is it an encoder or a decoder
    norm        = opt.norm # normalization
    use_dropout = opt.no_dropout # use dropout or not
    init_type   = opt.init_type # type of initialization
    init_gain   = opt.init_gain # initializing the gain

    norm_layer = get_norm_layer(norm_type=norm) # type of normalization used

    if net_type == 'resnet_encoder':
        n_blocks    = opt.resnet_blocks # number of resnet blocks (specified in params)
        net = ResnetEncoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
    elif net_type == 'resnet_decoder': 
        n_blocks    = opt.resnet_blocks
        # if n_blocks == 9:
        #     net = ResnetDecoder(input_nc, output_nc, ngf, linear_shape=opt_exp.input_shape, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, encoder_blocks=opt.encoder_res_blocks)
        # else:
        net = ResnetDecoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, encoder_blocks=opt.encoder_res_blocks)
        # create network block
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net_type)
    return init_net(net, init_type, init_gain, gpu_ids) # return the initialized network


def get_scheduler(optimizer, opt): # helper function to set the learning rate scheduler
    if opt.starting_epoch_count=='best' and opt.lr_policy == 'lambda': # if we want to record the best epoch AND the learning rate update rule is the lambda rule
        def lambda_rule(epoch):
            print("lambda update %s, %s, %s", (epoch, 0))
            lr_l = 1.0 - max(0, epoch + 1 - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) # learning rate update happens as initial_lr * lambda_rule(epoch)
    elif opt.starting_epoch_count!='best' and opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            print("lambda update %s, %s, %s", (epoch, opt.starting_epoch_count))
            lr_l = 1.0 - max(0, epoch + 1 + opt.starting_epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) # learning rate update happens as initial_lr * lambda_rule(epoch)
    # other learning rate update rules
    elif opt.starting_epoch_count!='best' and opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.9)
    elif opt.starting_epoch_count!='best' and opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.starting_epoch_count!='best' and opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler # learning rate update rule


def get_norm_layer(norm_type='instance'): # helper function; setting the type of normalization to be used
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True) # set batchnorm, with affine parameter = True by default
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False) # set instancenorm, with affine = False by default
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', gain=1): # helper function to initialize the network weights
    # m.weight.data is the weights of the given module
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func) # .apply() in nn.Module is used to initialize weights, for each layer


def init_net(net, init_type='normal', init_gain=1, gpu_ids=[]): # helper function which initializes weights and decides compute type
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        device_ = torch.device('cuda:{}'.format(gpu_ids[0]))
        # device_ = torch.device('cpu')
#         net.to(d)
        gpu_ids_int = list(map(int,gpu_ids))
        net = torch.nn.DataParallel(net, gpu_ids_int) # data parallelism, splits data across the given GPUs
        net.to(device_) # specifies the device for the network
    init_weights(net, init_type, gain=init_gain) # initialize network weights
    return net

def localization_error(output_predictions,input_labels,scale=1): # computes the localization error
    """
    output_predictions: (N,1,H,W), model prediction 
    input_labels: (N,1,H,W), ground truth target
    """
    input_labels[:,0] *= opt_exp.xscale
    output_predictions[:,0] *= opt_exp.xscale
    input_labels[:,1] *= opt_exp.yscale
    output_predictions[:,1] *= opt_exp.yscale

    image_size = output_predictions.shape
    error = np.zeros(image_size[0])

    for i in range(image_size[0]):
        label_temp = input_labels[i,:] # ground truth label
        pred_temp = output_predictions[i,:] # model prediction

        # label_index = np.asarray(np.unravel_index(np.argmax(label_temp), label_temp.shape))
        # pred_index = np.asarray(np.unravel_index(np.argmax(pred_temp),pred_temp.shape))
        
        error[i] = np.sqrt( np.sum( np.power(np.multiply( label_temp-pred_temp, scale ), 2)) )
    
    return error

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def RGB2Gray(img):
    return 0.2125*img[:,:,0] + 0.7154*img[:,:,1] + 0.0721*img[:,:,2]
