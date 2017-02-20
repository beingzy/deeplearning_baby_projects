""" training a Generative Adversarial Networks (GAN) with pytorch
    tutorial url: http://bit.ly/2lko6Wt

    Author: Yi Zhang <beingzy@gmail.com>
    Date: 2017/02/19
"""
import os
import datetime
import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.autograd import Variable


## ===============
## Data Params
## ===============
data_mean = 4 
data_stddev = 1.25 


## ===============
## Model Params
## ===============
g_input_size  = 1   # random noise dimension coming into generator, per output vector 
g_hidden_size = 50  # Generator complexity 
g_output_size = 1   # size of generated output vector 
g_input_size  = 100 # minibatch size - cardinality 
g_hidden_size = 50  # discriminator complexity 

d_input_size  = 100
d_hidden_size = 50
d_output_size = 1   # single dimension for 'real' vs. 'fake'

minibatch_size = d_input_size 

d_learning_rate = 2e-4 
g_learning_rate = 2e-4 
optim_betas = (0.9, 0.999)
num_epochs = 30000
print_inverval  = 200
# 'k' steps in the original GAN paper. Can put the discriminator on 
# higher training freq than generator
d_steps = 1 
g_steps = 1 

##
# (name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
(name, preprocess, d_input_func) = ("Data and Variance", 
    lambda data: decorate_with_diffs(data, 2.0), 
    lambda x: x * 2)

print("Using data {}".format(name))


## ==========================================
## Data: target data and generator input data
## ==========================================
def get_distribution_sampler(mu, sigma):
    #def output_func(n):
    #   return torch.Tensor(np.random.normal(mu, sigma), (1, n))
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))
    #return output_func


def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)

## ===============================================
## MODELS: Generator model and Discriminator model 
## ===============================================
class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.sigmoid(self.map2(x))
        return self.map3(x)


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))


def extract(v):
    return v.data.storage().tolist() 

def stats(d):
    return [np.mean(d), np.std(d)]

def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)



## ====================
## loop: alternating training Discriminator vs. Generator
## ====================
d_sampler = get_distribution_sampler(data_mean, data_stddev)
gi_sampler = get_generator_input_sampler()

G = Generator(input_size=g_input_size, 
    hidden_size=g_hidden_size, output_size=g_output_size)

D = Discriminator(input_size=d_input_func(d_input_size), 
    hidden_size=d_hidden_size, output_size=d_output_size)

criterion = nn.BCELoss() # Binary cross Entropy: http://pytorch.org/docs/nn.html#bceloss 
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

res_records = []
for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # 1. train D and real+fake
        D.zero_grad()

        # 1A: Train D on real 
        d_real_data = Variable(d_sampler(d_input_size))
        d_real_decision = D(preprocess(d_real_data))
        d_real_error = criterion(d_real_decision, Variable(torch.ones(1))) # ones = True 
        d_real_error.backward() # compute/store gradients, but do not change params 

        # 1B: Train D on fake 
        d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        d_fake_data = G(d_gen_input).detach() # detach to avoid training G on these labels 
        d_fake_decision = D(preprocess(d_fake_data.t()))
        d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1))) # zeros = fake 
        d_fake_error.backward()
        d_optimizer.step() 

    for g_index in range(g_steps):
        # 2. train G and D's response (but DO NOT train D on these labels)
        G.zero_grad()

        gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        g_fake_data = G(gen_input)
        dg_fake_decision = D(preprocess(g_fake_data.t()))
        # we want to foll, so pretend it's all genuine
        g_error = criterion(dg_fake_decision, Variable(torch.ones(1))) 

        g_error.backward() 
        g_optimizer.step() # only optimize G's parameters

    if epoch % print_inverval == 0:
        print("{}: D: {}/{} G: {} (Real: {}, Fake: {})".format(
            epoch, extract(d_real_error)[0], extract(d_fake_error)[0], extract(g_error)[0],
            stats(extract(d_real_data)), stats(extract(d_fake_data)))
        )
        record = {"epoch": epoch, "d_real_error": extract(d_real_error)[0],
         "d_fake_error": extract(d_fake_error)[0], 'g_error': extract(g_error)[0], 
         'mean_d_real_data': stats(extract(d_real_data))[0], 
         'std_d_real_data': stats(extract(d_real_data))[1], 
         'mean_d_fake_data': stats(extract(d_fake_data))[0], 
         'std_d_fake_data': stats(extract(d_fake_data))[1]}

        res_records.append(record)

outfile_path = os.path.join(os.getcwd(), "training_log_{}.txt".format('2017feb18'))
pd.DataFrame(res_records).to_csv(outfile_path, sep=',', encoding='utf-8')
print("traing record is exported: {}".format(outfile_path))



