require 'image'
require 'torch'
require 'nn'

print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- data:
cmd:option('-size', 'full', 'how many samples do we load: small | full | extra')
-- model:
cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
-- loss:
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-8, 'learning rate at t=0 used as 1e-3 for all fine tunnings ')
--cmd:option('-learningRate', 1e-4, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-states', (5*4096), '2 is number of classifiers')
cmd:option('-classes', 5, 'Number of classes')
--cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:text()
opt = cmd:parse(arg or {})
---------------------------------------------------------------------------
-- nb of threads and fixed seed (for repeatable experiments)
torch.setdefaulttensortype('torch.FloatTensor')

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)



dofile 'big2.lua'

dofile 'training.lua'
dofile 'testing.lua'

while(1) do
train()
   test()
  print(mean,std)
    -- print(mean1,std1)
end