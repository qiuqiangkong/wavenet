import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
import librosa
from utilities import write_audio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import config
import mu_law
from models import move_data_to_gpu, WaveNet
from datasets import *

from utilities import create_folder, get_filename, create_logging, read_audio


def _loss_func(output, target):
    '''Loss function. 
    
    Args:
      output: (batch_size, seq_len, quantize_bins)
      target: (batch_size, seq_len, quantize_bins)
    '''
    loss = F.nll_loss(
        output.view(-1, output.shape[-1]), target.view(target.shape[-1]))
        
    return loss


def evaluate(model, validate_loader, condition, cuda):
    '''Evaluate the loss of partial data during training. 
    '''
    
    losses = []
    
    for (iteration, (batch_x, global_condition)) in enumerate(validate_loader):
        
        if condition:
            global_condition = move_data_to_gpu(global_condition, cuda)
        else:
            global_condition = None
        
        batch_x = move_data_to_gpu(batch_x, cuda)
        
        batch_input = batch_x[:, 0 : -1]
        output_width = batch_input.shape[-1] - model.receptive_field + 1
        batch_target = batch_x[:, - output_width :]
            
        with torch.no_grad():
            model.eval()
            batch_output = model(batch_input, global_condition)
            loss = _loss_func(batch_output, batch_target)
            losses.append(loss.data.cpu().numpy())

        if iteration == 100:
            break
            
    return np.mean(losses)
        

def train(args):

    # Arugments & parameters
    dataset = args.dataset
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    filename = args.filename
    condition = args.condition
    cuda = args.cuda
    
    batch_size = 1  # Use an audio clip as a mini-batch. Must be 1 if audio 
                    # clips has different length. 
    quantize_bins = config.quantize_bins
    dilations = config.dilations
    
    # Paths
    models_dir = os.path.join(workspace, 'models', 'dataset={}'.format(dataset), 
                              filename, 'condition={}'.format(condition))
    create_folder(models_dir)

    # Data Generator
    Dataset = get_dataset('vctk')
    
    train_dataset = Dataset(dataset_dir, data_type='train')    
    validate_dataset = VCTKDataset(dataset_dir, data_type='validate')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    
    validate_loader = torch.utils.data.DataLoader(validate_dataset, 
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    # Model
    model = WaveNet(dilations, 
                    residual_channels=config.residual_channels, 
                    dilation_channels=config.dilation_channels, 
                    skip_channels=config.skip_channels, 
                    quantize_bins=config.quantize_bins, 
                    global_condition_channels=config.global_condition_channels, 
                    global_condition_cardinality=config.global_condition_cardinality, 
                    use_cuda=cuda)

    if cuda:
        model.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0.)

    train_bgn_time = time.time()

    while True:
        for (iteration, (batch_x, global_condition)) in enumerate(train_loader):
            '''batch_x: (batch_size, seq_len)
            global_condition: (batch_size,)
            '''

            print('iteration: {}, input size: {}'.format(iteration, batch_x.shape))
            
            # Evaluate
            if iteration % 1000 == 0:
                train_fin_time = time.time()
                evaluate_bgn_time = time.time()
                loss = evaluate(model, validate_loader, condition, cuda)
                
                print('-----------------')
                logging.info('iteration: {}, loss: {:.3f}, train_time: {:.3f}, '
                    'validate time: {:.3f} s'.format(iteration, loss, 
                    train_fin_time - train_bgn_time, time.time() - evaluate_bgn_time))
                    
                train_bgn_time = time.time()
            
            # Save model
            if iteration % 10000 == 0:
                save_out_dict = {'iteration': iteration, 
                                'state_dict': model.state_dict(), 
                                'optimizer': optimizer.state_dict()}
                                
                save_out_path = os.path.join(models_dir, 
                    'md_{}_iters.tar'.format(iteration))
                    
                torch.save(save_out_dict, save_out_path)
                logging.info('Save model to {}'.format(save_out_path))
            
            # Move data to GPU
            if condition:
                global_condition = move_data_to_gpu(global_condition, cuda)
            else:
                global_condition = None
            
            batch_x = move_data_to_gpu(batch_x, cuda)
            
            # Prepare input and target data
            batch_input = batch_x[:, 0 : -1]
            output_width = batch_input.shape[-1] - model.receptive_field + 1
            batch_target = batch_x[:, - output_width :]

            # Forward
            model.train()
            batch_output = model(batch_input, global_condition)
            loss = _loss_func(batch_output, batch_target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print('loss: {:.3f}'.format(loss.data.cpu().numpy()))
            
     
def generate(args):
    
    np.random.seed(1234)
    
    # Arugments & parameters
    workspace = args.workspace
    iteration = args.iteration
    samples = args.samples
    global_condition = args.global_condition
    fast_generate = args.fast_generate
    cuda = args.cuda
    filename = args.filename
    
    batch_size = 1
    quantize_bins = config.quantize_bins
    dilations = config.dilations
    
    if global_condition == -1:
        condition = False
        global_condition = None
    else:
        condition = True
        global_condition = torch.LongTensor([global_condition] * batch_size)
        if cuda:
            global_condition = global_condition.cuda()
    
    # Paths
    model_path = os.path.join(workspace, 'models', 'dataset={}'.format(dataset), 
                              filename, 'condition={}'.format(condition), 
                              'md_{}_iters.tar'.format(iteration))
                              
    generated_wavs_dir = os.path.join(workspace, 'generated_wavs', filename, 
                                      'condition={}'.format(condition))
                                      
    create_folder(generated_wavs_dir)
    
    # Load model
    model = WaveNet(dilations, 
                    residual_channels=config.residual_channels, 
                    dilation_channels=config.dilation_channels, 
                    skip_channels=config.skip_channels, 
                    quantize_bins=config.quantize_bins, 
                    global_condition_channels=config.global_condition_channels, 
                    global_condition_cardinality=config.global_condition_cardinality, 
                    use_cuda=cuda)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    if cuda:
        model.cuda()
    
    receptive_field = model.receptive_field
        
    # Init buffer for generation
    _mulaw = mu_law.MuLaw(mu=quantize_bins)
    _quantize = mu_law.Quantize(quantize=quantize_bins)
    
    buffer = np.zeros((batch_size, receptive_field))
    buffer[:, -1] = np.random.uniform(-1, 1, batch_size)
    buffer = _mulaw.transform(buffer)
    buffer = _quantize.transform(buffer)
    buffer = torch.LongTensor(buffer)
    
    if cuda:
        buffer = buffer.cuda()
    
    # Generate
    generate_time = time.time()
    
    if fast_generate:
        with torch.no_grad():
            model.eval()
            audio = model.fast_generate(buffer=buffer, 
                                        samples=samples, 
                                        global_condition=global_condition)
        
    else:
        with torch.no_grad():
            model.eval()
            audio = model.slow_generate(buffer=buffer, 
                                        global_condition=global_condition, 
                                        samples=samples)
    
    print('Generate_time: {:.3f} s'.format(time.time() - generate_time))
    
    audio = audio.data.cpu().numpy()

    # Transform to wave
    audio = audio[0]
    audio = _quantize.inverse_transform(audio)
    audio = _mulaw.inverse_transform(audio)
        
    audio_path = os.path.join(generated_wavs_dir, 'iter_{}.wav'.format(iteration))
    write_audio(audio_path, audio, config.sample_rate)
    print('Generate wav to {}'.format(audio_path))
    
    plt.plot(audio)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset', type=str, required=True)
    parser_train.add_argument('--dataset_dir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--condition', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    parser_generate = subparsers.add_parser('generate')
    parser_generate.add_argument('--dataset', type=str, required=True)
    parser_generate.add_argument('--workspace', type=str, required=True)
    parser_generate.add_argument('--iteration', type=int, required=True)
    parser_generate.add_argument('--samples', type=int, required=True)
    parser_generate.add_argument('--global_condition', type=int, required=True)
    parser_generate.add_argument('--fast_generate', action='store_true', default=False)
    parser_generate.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'generate':
        generate(args)

    else:
        raise Exception('Error argument!')