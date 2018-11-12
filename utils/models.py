import math
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


def move_data_to_gpu(x, cuda):
    if cuda:
        x = x.cuda()
    return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    if layer.weight.ndimension() == 3:
        nn.init.xavier_uniform_(layer.weight)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)
    
def sparse_to_categorical(y, n_out, cuda):
    n_samples = len(y)
    y_mat = torch.zeros(n_samples, n_out)
    
    if cuda:
        y_mat = y_mat.cuda()
    
    y = y.view(n_samples, 1)
    y_mat.scatter_(1, y, 1)
    return y_mat

        
class WaveNet(nn.Module):
    def __init__(self, 
                 dilations, 
                 residual_channels=32, 
                 dilation_channels=32, 
                 skip_channels=512, 
                 quantize_bins=256, 
                 global_condition_channels=32, 
                 global_condition_cardinality=377, 
                 use_cuda=True):
                     
        super(WaveNet, self).__init__()
        
        self.dilations = dilations
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.quantize_bins = quantize_bins
        self.global_condition_channels = global_condition_channels
        self.global_condition_cardinality = global_condition_cardinality
        self.use_cuda = use_cuda
        
        self.receptive_field = sum(dilations) + 2
        
        self.dict = {}
        
        # Condition embedding layer
        self.dict['gc_embedding'] = nn.Embedding(
            num_embeddings=self.global_condition_cardinality, 
            embedding_dim=self.global_condition_channels)
            
        self.add_module(name='gc_embedding', module=self.dict['gc_embedding'])
        
        # Casual layer
        self.dict['causal_layer'] = nn.Conv1d(
            in_channels=quantize_bins, out_channels=self.residual_channels, 
            kernel_size=2, stride=1, padding=0, dilation=1, bias=True)
            
        self.add_module(name='causal_layer', module=self.dict['causal_layer'])
        init_layer(self.dict['causal_layer'])
        
        # Dilated layers
        self.dict['dilated_stack'] = []
        
        for i, dilation in enumerate(dilations):
            
            current = {}
            
            # Residual connections
            current['filter'] = nn.Conv1d(in_channels=self.residual_channels, 
                out_channels=self.dilation_channels, kernel_size=2, stride=1, 
                padding=0, dilation=dilation, bias=True)
                
            current['gate'] = nn.Conv1d(in_channels=self.residual_channels, 
                out_channels=self.dilation_channels, kernel_size=2, stride=1, 
                padding=0, dilation=dilation, bias=True)
                
            current['dense'] = nn.Conv1d(in_channels=self.dilation_channels, 
                out_channels=self.residual_channels, kernel_size=1, stride=1, 
                padding=0, dilation=1, bias=True)
                
            current['skip'] = nn.Conv1d(in_channels=self.dilation_channels, 
                out_channels=self.skip_channels, kernel_size=1, stride=1, 
                padding=0, dilation=1, bias=True)

            init_layer(current['filter'])
            init_layer(current['gate'])
            init_layer(current['dense'])
            init_layer(current['skip'])

            self.add_module(name='dilated_{}_filter'.format(i), module=current['filter'])
            self.add_module(name='dilated_{}_gate'.format(i), module=current['gate'])
            self.add_module(name='dilated_{}_dense'.format(i), module=current['dense'])
            self.add_module(name='dilated_{}_skip'.format(i), module=current['skip'])
            
            # Condition parameters
            current['gc_filtweights'] = nn.Conv1d(
                in_channels=self.global_condition_channels, 
                out_channels=self.dilation_channels, kernel_size=1, 
                stride=1, padding=0, dilation=dilation, bias=False)
                
            current['gc_gateweights'] = nn.Conv1d(
                in_channels=self.global_condition_channels, 
                out_channels=self.dilation_channels, kernel_size=1, stride=1, 
                padding=0, dilation=dilation, bias=False)
            
            init_layer(current['gc_filtweights'])
            init_layer(current['gc_gateweights'])
            
            self.add_module(name='dilated_{}_gc_filtweights'.format(i), module=current['gc_filtweights'])
            self.add_module(name='dilated_{}_gc_gateweights'.format(i), module=current['gc_gateweights'])
            
            self.dict['dilated_stack'].append(current)
            
        # Fully connected layers
        self.dict['postprocess1'] = nn.Conv1d(in_channels=self.skip_channels, 
            out_channels=self.skip_channels, kernel_size=1, stride=1, 
            padding=0, dilation=1, bias=True)
            
        self.dict['postprocess2'] = nn.Conv1d(in_channels=self.skip_channels, 
            out_channels=self.quantize_bins, kernel_size=1, stride=1, padding=0, 
            dilation=1, bias=True)
        
        init_layer(self.dict['postprocess1'])
        init_layer(self.dict['postprocess2'])
        
        self.add_module(name='postprocess1', module=self.dict['postprocess1'])
        self.add_module(name='postprocess2', module=self.dict['postprocess2'])
            
    def forward(self, input, global_condition=None, queues=None):
        '''Forward data to a WaveNet model. 
        
        Args:
          input: (batch_size, seq_len), int
          global_condition: (batch_size,), int
        '''
        
        (batch_size, seq_len) = input.shape
        quantize_bins = self.quantize_bins
        
        # One-hot encoding
        x = sparse_to_categorical(input.view(-1), quantize_bins, self.use_cuda)
        x = x.view((batch_size, seq_len, quantize_bins))
        x = x.transpose(1, 2)   
        '''(batch_size, quantize_bins, seq_len)'''
        
        # Condition embedding
        if global_condition is None:
            gc_embedding = None
        else:
            gc_embedding = self.dict['gc_embedding'](global_condition) # (batch_size, gc_channels)
            gc_embedding = gc_embedding[:, :, None]
        
        output_width = input.shape[-1] - self.receptive_field + 1
        
        # Causal convolution
        x = self.dict['causal_layer'](x)
        
        # Outputs from all layers
        skip_contributions = []
        
        for layer_index, dilation in enumerate(self.dilations):
            
            # Queues are only used to initialize the cache in fast generation. 
            if queues is not None:
                queues[layer_index].data = x[:, :, - dilation - 1 : -1]
            
            (skip_contribution, x) = self._create_dilation_layer(x, layer_index, dilation, 
                gc_embedding, output_width)
                
            skip_contributions.append(skip_contribution)
            
        total = sum(skip_contributions)
  
        x = F.relu(total)
        x = F.relu(self.dict['postprocess1'](x))
        x = self.dict['postprocess2'](x)
        x = F.log_softmax(x.transpose(1, 2), dim=-1)        
        
        if queues is not None:
            return x, queues
        else:
            return x
        
    def _create_dilation_layer(self, input, layer_index, dilation, 
        gc_embedding, output_width):
        '''Forward in a residual block. 
        
        Args:
          input: (batch_size, residual_channels, seq_len)
          layer_index: int
          dilation: int
          gc_embedding: (batch_size, global_condition_channels, 1)
          output_width: int
        '''
        conv_filter = self.dict['dilated_stack'][layer_index]['filter'](input)
        conv_gate = self.dict['dilated_stack'][layer_index]['gate'](input)

        if gc_embedding is not None:
            
            conv_filter += self.dict['dilated_stack'][layer_index]['gc_filtweights'](gc_embedding)
            conv_gate += self.dict['dilated_stack'][layer_index]['gc_gateweights'](gc_embedding)
            
        out = F.tanh(conv_filter) * F.sigmoid(conv_gate)
        
        transformed = self.dict['dilated_stack'][layer_index]['dense'](out)
        transformed = input[:, :, -transformed.shape[-1] :] + transformed
        
        skip_contribution = self.dict['dilated_stack'][layer_index]['skip'](out[:, :, - output_width :])
        
        return skip_contribution, transformed
        
    def slow_generate(self, buffer, samples, global_condition):
        '''Vanilla WaveNet generation. Time complexity: O(2^layers_num)
        
        Args:
          buffer: (batch_size, seq_len), int
          samples: int, number of samples to generate
          global_condition: (batch_size,), int | None
        '''
        (batch_size, _) = buffer.shape
        receptive_field = self.receptive_field
        
        for t in tqdm(range(samples)):
            
            log_prob = self.forward(input=buffer[:, -receptive_field :], 
                global_condition=global_condition, queues=None)
            '''(batch_size, 1, quantize_bins)'''
            
            prob = torch.exp(log_prob[0, 0])
            next_sample = torch.multinomial(input=prob, num_samples=1, replacement=True)
            buffer = torch.cat((buffer, next_sample.view(batch_size, 1)), dim=-1)
            
        return buffer
            
        
    def _fast_create_dilation_layer(self, input_batch, layer_index, gc_embedding):
        '''Forward in a residual block using cache. 
        
        Args:
          input: (batch_size, residual_channels, 2)
          layer_index: int
          dilation: int
          gc_embedding: (batch_size, global_condition_channels, 1)
          output_width: int
        '''
        
        conv_filter = self.dilated_conv(self.dict['dilated_stack'][layer_index]['filter'], input_batch)
        conv_gate = self.dilated_conv(self.dict['dilated_stack'][layer_index]['gate'], input_batch)
        
        if gc_embedding is not None:
            
            conv_filter += self.dict['dilated_stack'][layer_index]['gc_filtweights'](gc_embedding)
            conv_gate += self.dict['dilated_stack'][layer_index]['gc_gateweights'](gc_embedding)
        
        out = F.tanh(conv_filter) * F.sigmoid(conv_gate)
        
        transformed = self.dict['dilated_stack'][layer_index]['dense'](out)
        
        skip_contribution = self.dict['dilated_stack'][layer_index]['skip'](out)
        
        return skip_contribution, transformed + input_batch[:, :, -1:]
        
    def fast_generate(self, buffer, samples, global_condition):
        '''Fast generation using cache. Time complexity: O(layers_num). Using
        CPU is faster than GPU. +100% faster than slow_generate. 
        
        Args:
          buffer: (batch_size, seq_len)
          samples: int
          global_condition: (batch_size,), int | None
        '''

        (batch_size, _) = buffer.shape
        quantize_bins = self.quantize_bins
        
        # Init queues for fast generation
        queues = []     
        
        for (layer_index, dilation) in enumerate(self.dilations):
            
            queue = Queue(batch_size=batch_size, 
                num_channels=self.residual_channels, max_length=dilation, 
                use_cuda=self.use_cuda)
                
            queues.append(queue)
        
        # Init cache
        (_, queues) = self.forward(input=buffer, 
            global_condition=global_condition, queues=queues)
    
        # Condition
        if global_condition is None:
            gc_embedding = None
        else:
            gc_embedding = self.dict['gc_embedding'](global_condition) 
            gc_embedding = gc_embedding[:, :, None]
            '''(batch_size, gc_channels, seq_len)'''
       
        for t in tqdm(range(samples)):
            
            x = buffer[:, -2:]  # Fast generation only need to feed last 2 samples
            x = sparse_to_categorical(x.view(-1), quantize_bins, self.use_cuda)
            x = x.view((batch_size, 2, quantize_bins))
            x = x.transpose(1, 2)   
            '''(batch_size, quantize_bins, seq_len)'''
            
            # Casual layer
            x = self.dict['causal_layer'](x)
            
            skip_contributions = []
            
            for (layer_index, dilation) in enumerate(self.dilations):
                
                # Cache
                pad = queues[layer_index].dequeue()
                queues[layer_index].enqueue(x)
                
                x = torch.cat((pad, x), dim=-1)
                
                # Forward in a residual using cache
                (skip_contribution, x) = self._fast_create_dilation_layer(
                    input_batch=x, layer_index=layer_index, gc_embedding=gc_embedding)
                
                skip_contributions.append(skip_contribution)
                
            total = sum(skip_contributions)
    
            x = F.relu(total)
            x = F.relu(self.dict['postprocess1'](x))
            x = self.dict['postprocess2'](x)
            x = F.log_softmax(x.transpose(1, 2), dim=-1) 

            log_prob = torch.exp(x)
        
            next_sample = torch.multinomial(input=log_prob[0, 0], num_samples=1, replacement=True)
            next_sample = next_sample[:, None]
            buffer = torch.cat((buffer, next_sample), dim=-1)
        
        return buffer
        
    def dilated_conv(self, layer, x):
        dilation = layer.dilation
        layer.dilation = (1,)
        out = layer(x)
        layer.dilation = dilation
        return out
        
        
class Queue(object):
    def __init__(self, batch_size, num_channels, max_length, use_cuda):
        self.data = torch.Tensor(batch_size, num_channels, max_length)
        if use_cuda:
            self.data = self.data.cuda()
        
        self.reset()
        
    def enqueue(self, input):
        if self.data.shape[-1] > 1:
            self.data[:, :, 0 : -1] = self.data[:, :, 1 : ]
        
        self.data[:, :, -1:] = input
        
    def dequeue(self):
        return self.data[:, :, 0][:, :, None].clone()
        
    def reset(self):
        self.data.zero_()