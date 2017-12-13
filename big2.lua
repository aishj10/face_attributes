require 'image'
require 'torch'
require 'nn'


-- parse command line arguments

---------------------------------data_train.lua-----------------------


----------------------------------------------------------------------
print '==> executing all'

----------------------------------------------------------------------
print '==> downloading dataset'


--local size=1200 --top 30 classes
--local size=13231 --all classes
 --size=1456  --top 10 classes

 size=272 ---5 classes
 trsize = 222 
tesize = 50       

--size=5 ---5 classes
-- trsize = 3 
--tesize = 2    


local imagesAll = torch.Tensor(size,3,250,250):zero()
 local labelsAll = torch.Tensor(size):zero()
local images = torch.Tensor(size,3,250,250):zero()


--filename='/home/aishwarya/desktop/test_codes/list_sort'
filename='./list'
print '==> Read LFW Dataset'
                                                                                                                                                                                                                                                                                              
local file = assert(io.open(filename, "r"))
local f=1
local lab=1

while true do
      local s = file:read("*line")
    if  (f>size) or (not s) then break end 

  n1,n2=string.match(s, '(%S+)%s*(%d+)')
  n2=tonumber(n2)
  --print('n2',n2)
  
  if n2 < 2 then
    --print(f)
    images[f] = image.load('/home/aishwarya/desktop/lfw/'..n1..'/'..n1..'_0001.jpg',3,'float')
    
    labelsAll[f] =lab
    --print(n1,labelsAll[f])
    
    lab=lab+1
    f=f+1
  
  
  else
    
    for t=1, n2 do
    --print(f)
    if t >=1  and t <= 9 then 
    images[f] = image.load('/home/aishwarya/desktop/lfw/'..n1..'/'..n1..'_000'..t..'.jpg',3,'float')
    
    elseif t >=10 and t <= 99 then
    images[f] = image.load('/home/aishwarya/desktop/lfw/'..n1..'/'..n1..'_00'..t..'.jpg',3,'float')
           elseif t >=100 and t <= 999 then
    images[f] = image.load('/home/aishwarya/desktop/lfw/'..n1..'/'..n1..'_0'..t..'.jpg',3,'float')
    end
    
    labelsAll[f] =lab
    --print(n1,labelsAll[f])
    f=f+1
    
    end
    lab=lab+1
  end


end




print '==> Rescale Images'
mean = {129.1863,104.7624,93.5940}
for i=1,size do

local im = images[i]
im=im*255
local im_bgr = im:index(1,torch.LongTensor{3,2,1})
for j=1,3 do im_bgr[j]:add(-mean[j]) end
imagesAll[i] =im_bgr

end


----------------------------End of data_train.lua----------------------------------------
------------------------------vgg.lua-----------------------------------

local threads = 4
local offset  = 0

--torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(threads)
print('==> #threads:', torch.getnumthreads())
torch.setdefaulttensortype('torch.FloatTensor')


--net = torch.load('/home/aishwarya/desktop/test_codes/results/model_random')
local net = torch.load('/home/aishwarya/desktop/open/vgg_face_torch/VGG_FACE.t7')
local model1= nn.Sequential()
print('==>Print model')
n=net.modules
i=1
--while(i<=18) do
  while(i<=36) do
model1:add(n[i])
i=i+1
end
--print(model)
 model1:evaluate()




print('==> Pass Through model')

local sc={256, 384 , 512}
local img_sc = torch.Tensor(10,3,224,224):zero()
local out_sc=torch.Tensor(10,4096):zero()
local out_avg=torch.Tensor(3,4096):zero()

local vgg=torch.Tensor(size,4096):zero()

t=1
k=1
while (t<=size)  do
  for i=1,3 do

      img = image.scale(imagesAll[t],sc[i],sc[i])
      iw = img:size(3)
       ih = img:size(2)
      
           img_sc[1]=image.crop(img,0,0,224,224)        
           img_sc[2]=image.crop(img,iw-224,0,iw,224)
           img_sc[3]=image.crop(img,0,ih-224,224,ih)
            img_sc[4]=image.crop(img,iw-224,ih-224,iw,ih)
           img_sc[5]=image.crop(img,iw/2-112,ih/2-112,iw/2+112,ih/2+112)
            img_sc[6]=image.hflip(img_sc[1])
            img_sc[7]=image.hflip(img_sc[2])
            img_sc[8]=image.hflip(img_sc[3])
            img_sc[9]=image.hflip(img_sc[4])
            img_sc[10]=image.hflip(img_sc[5])

          --  print('==> Pass Through model------1')

            for j=1,10 do
              out_sc[j]=model1:forward(img_sc[j])
            end
            out_avg[i]=torch.sum(out_sc,1)
          end

       --   print('==> Take Average')

      output1=torch.sum(out_avg,1)
      output1=output1/30

      -------------------Normalise-----------------------------
         local x=torch.max(output1)

        output1=torch.div(output1,x)

       local norm=torch.norm(output1)
            output1=torch.div(output1,norm)

      vgg[t]=output1




      

k=k+1
t=t+1
end

--for i=1,size do
--indata.data[i]=vgg[i]

--end
torch.save('vgg',vgg)
torch.save('labelsAll',labelsAll)
----------------------------End of vgg.lua----------------------------
--------------------------gen.lua-----------------------------------------

local threads = 4
local offset  = 0

--torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(threads)
print('==> #threads:', torch.getnumthreads())
torch.setdefaulttensortype('torch.FloatTensor')

--net = torch.load('/home/aishwarya/desktop/test_codes/results/model_random')
local net = torch.load('/home/aishwarya/desktop/att/genfull_results/model')
local model1= nn.Sequential()
print('==>Print model')
n=net.modules
i=1
--while(i<=18) do
  while(i<=36) do
model1:add(n[i])
i=i+1
end
--print(model)
----------------------------------test model--------------------------------------
local model_test=nn.Sequential()

model_test:add(n[37])
model_test:add(n[38])
model_test:add(n[39])


------------------------------------------------------------------------------------


 model1:evaluate()


print('==> Pass Through model')

local sc={256, 384 , 512}
local img_sc = torch.Tensor(10,3,224,224):zero()
local out_sc=torch.Tensor(10,4096):zero()
local out_avg=torch.Tensor(3,4096):zero()

local gen=torch.Tensor(size,4096):zero()

t=1
k=1
while (t<=size)  do
  for i=1,3 do

      img = image.scale(imagesAll[t],sc[i],sc[i])
      iw = img:size(3)
       ih = img:size(2)
      
           img_sc[1]=image.crop(img,0,0,224,224)        
           img_sc[2]=image.crop(img,iw-224,0,iw,224)
           img_sc[3]=image.crop(img,0,ih-224,224,ih)
            img_sc[4]=image.crop(img,iw-224,ih-224,iw,ih)
           img_sc[5]=image.crop(img,iw/2-112,ih/2-112,iw/2+112,ih/2+112)
            img_sc[6]=image.hflip(img_sc[1])
            img_sc[7]=image.hflip(img_sc[2])
            img_sc[8]=image.hflip(img_sc[3])
            img_sc[9]=image.hflip(img_sc[4])
            img_sc[10]=image.hflip(img_sc[5])

          --  print('==> Pass Through model------1')

            for j=1,10 do
              out_sc[j]=model1:forward(img_sc[j])
            end
            out_avg[i]=torch.sum(out_sc,1)
          end

       --   print('==> Take Average')

      output1=torch.sum(out_avg,1)
      output1=output1/30

      -------------------Normalise-----------------------------
         local x=torch.max(output1)

        output1=torch.div(output1,x)

       local norm=torch.norm(output1)
            output1=torch.div(output1,norm)

      gen[t]=output1




          


k=k+1
t=t+1
end

torch.save('gen',gen)

vgg=torch.cat(vgg,gen)

--indata.data=torch.cat(indata.data,gen,2)
--print('IN_GEN DATA')
--print(gen[1][1],gen[2][1])
----------------------------------------test model----------------------------------------------
--for t=1,size do
 -- local prob =model_test:forward(gen[t])
 -- maxval,maxid = prob:max(1)
 -- print(maxid[1])
--end

----------------------------------End of gen.lua----------------------------------------------

--------------------------blondhair.lua-----------------------------------------

local threads = 4
local offset  = 0

--torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(threads)
print('==> #threads:', torch.getnumthreads())
torch.setdefaulttensortype('torch.FloatTensor')

--net = torch.load('/home/aishwarya/desktop/test_codes/results/model_random')
local net = torch.load('/home/aishwarya/desktop/att/blondhair_results/model')
local model1= nn.Sequential()
print('==>Print model')
n=net.modules
i=1
--while(i<=18) do
  while(i<=36) do
model1:add(n[i])
i=i+1
end
--print(model)
----------------------------------test model--------------------------------------
--local model_test=nn.Sequential()

--model_test:add(n[37])
--model_test:add(n[38])
--model_test:add(n[39])


------------------------------------------------------------------------------------
 model1:evaluate()




print('==> Pass Through model')

local sc={256, 384 , 512}
local img_sc = torch.Tensor(10,3,224,224):zero()
local out_sc=torch.Tensor(10,4096):zero()
local out_avg=torch.Tensor(3,4096):zero()

local blondhair=torch.Tensor(size,4096):zero()

t=1
k=1
while (t<=size)  do
  for i=1,3 do

      img = image.scale(imagesAll[t],sc[i],sc[i])
      iw = img:size(3)
       ih = img:size(2)
      
           img_sc[1]=image.crop(img,0,0,224,224)        
           img_sc[2]=image.crop(img,iw-224,0,iw,224)
           img_sc[3]=image.crop(img,0,ih-224,224,ih)
            img_sc[4]=image.crop(img,iw-224,ih-224,iw,ih)
           img_sc[5]=image.crop(img,iw/2-112,ih/2-112,iw/2+112,ih/2+112)
            img_sc[6]=image.hflip(img_sc[1])
            img_sc[7]=image.hflip(img_sc[2])
            img_sc[8]=image.hflip(img_sc[3])
            img_sc[9]=image.hflip(img_sc[4])
            img_sc[10]=image.hflip(img_sc[5])

          --  print('==> Pass Through model------1')

            for j=1,10 do
              out_sc[j]=model1:forward(img_sc[j])
            end
            out_avg[i]=torch.sum(out_sc,1)
          end

       --   print('==> Take Average')

      output1=torch.sum(out_avg,1)
      output1=output1/30

      -------------------Normalise-----------------------------
         local x=torch.max(output1)

        output1=torch.div(output1,x)

       local norm=torch.norm(output1)
            output1=torch.div(output1,norm)

      blondhair[t]=output1




          


k=k+1
t=t+1
end

torch.save('blondhair',blondhair)

vgg=torch.cat(vgg,blondhair)

--indata.data=torch.cat(indata.data,blondhair,2)
--print('IN_GEN DATA')
--print(gen[1][1],gen[2][1])
----------------------------------------test model----------------------------------------------
--for t=1,size do
 -- local prob =model_test:forward(gen[t])
 -- maxval,maxid = prob:max(1)
 -- print(maxid[1])
--end

----------------------------------End of blondhair.lua----------------------------------------------

--------------------------chubby.lua-----------------------------------------

local threads = 4
local offset  = 0

--torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(threads)
print('==> #threads:', torch.getnumthreads())
torch.setdefaulttensortype('torch.FloatTensor')

--net = torch.load('/home/aishwarya/desktop/test_codes/results/model_random')
local net = torch.load('/home/aishwarya/desktop/att/chubby_results/model')
local model1= nn.Sequential()
print('==>Print model')
n=net.modules
i=1
--while(i<=18) do
  while(i<=36) do
model1:add(n[i])
i=i+1
end
--print(model)
----------------------------------test model--------------------------------------
--local model_test=nn.Sequential()

--model_test:add(n[37])
--model_test:add(n[38])
--model_test:add(n[39])


------------------------------------------------------------------------------------

 model1:evaluate()



print('==> Pass Through model')

local sc={256, 384 , 512}
local img_sc = torch.Tensor(10,3,224,224):zero()
local out_sc=torch.Tensor(10,4096):zero()
local out_avg=torch.Tensor(3,4096):zero()

local chubby=torch.Tensor(size,4096):zero()

t=1
k=1
while (t<=size)  do
  for i=1,3 do

      img = image.scale(imagesAll[t],sc[i],sc[i])
      iw = img:size(3)
       ih = img:size(2)
      
           img_sc[1]=image.crop(img,0,0,224,224)        
           img_sc[2]=image.crop(img,iw-224,0,iw,224)
           img_sc[3]=image.crop(img,0,ih-224,224,ih)
            img_sc[4]=image.crop(img,iw-224,ih-224,iw,ih)
           img_sc[5]=image.crop(img,iw/2-112,ih/2-112,iw/2+112,ih/2+112)
            img_sc[6]=image.hflip(img_sc[1])
            img_sc[7]=image.hflip(img_sc[2])
            img_sc[8]=image.hflip(img_sc[3])
            img_sc[9]=image.hflip(img_sc[4])
            img_sc[10]=image.hflip(img_sc[5])

          --  print('==> Pass Through model------1')

            for j=1,10 do
              out_sc[j]=model1:forward(img_sc[j])
            end
            out_avg[i]=torch.sum(out_sc,1)
          end

       --   print('==> Take Average')

      output1=torch.sum(out_avg,1)
      output1=output1/30

      -------------------Normalise-----------------------------
         local x=torch.max(output1)

        output1=torch.div(output1,x)

       local norm=torch.norm(output1)
            output1=torch.div(output1,norm)

      chubby[t]=output1




          


k=k+1
t=t+1
end


torch.save('chubby',chubby)

vgg=torch.cat(vgg,chubby)
--indata.data=torch.cat(indata.data,chubby,2)
--print('IN_GEN DATA')
--print(gen[1][1],gen[2][1])
----------------------------------------test model----------------------------------------------
--for t=1,size do
 -- local prob =model_test:forward(gen[t])
 -- maxval,maxid = prob:max(1)
 -- print(maxid[1])
--end

----------------------------------End of blondhair.lua----------------------------------------------
--------------------------mustache.lua-----------------------------------------

local threads = 4
local offset  = 0

--torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(threads)
print('==> #threads:', torch.getnumthreads())
torch.setdefaulttensortype('torch.FloatTensor')

--net = torch.load('/home/aishwarya/desktop/test_codes/results/model_random')
local net = torch.load('/home/aishwarya/desktop/att/mustache_results/model')
local model1= nn.Sequential()
print('==>Print model')
n=net.modules
i=1
--while(i<=18) do
  while(i<=36) do
model1:add(n[i])
i=i+1
end
--print(model)
----------------------------------test model--------------------------------------
--local model_test=nn.Sequential()

--model_test:add(n[37])
--model_test:add(n[38])
--model_test:add(n[39])


------------------------------------------------------------------------------------

 model1:evaluate()



print('==> Pass Through model')

local sc={256, 384 , 512}
local img_sc = torch.Tensor(10,3,224,224):zero()
local out_sc=torch.Tensor(10,4096):zero()
local out_avg=torch.Tensor(3,4096):zero()

local mustache=torch.Tensor(size,4096):zero()

t=1
k=1
while (t<=size)  do
  for i=1,3 do

      img = image.scale(imagesAll[t],sc[i],sc[i])
      iw = img:size(3)
       ih = img:size(2)
      
           img_sc[1]=image.crop(img,0,0,224,224)        
           img_sc[2]=image.crop(img,iw-224,0,iw,224)
           img_sc[3]=image.crop(img,0,ih-224,224,ih)
            img_sc[4]=image.crop(img,iw-224,ih-224,iw,ih)
           img_sc[5]=image.crop(img,iw/2-112,ih/2-112,iw/2+112,ih/2+112)
            img_sc[6]=image.hflip(img_sc[1])
            img_sc[7]=image.hflip(img_sc[2])
            img_sc[8]=image.hflip(img_sc[3])
            img_sc[9]=image.hflip(img_sc[4])
            img_sc[10]=image.hflip(img_sc[5])

          --  print('==> Pass Through model------1')

            for j=1,10 do
              out_sc[j]=model1:forward(img_sc[j])
            end
            out_avg[i]=torch.sum(out_sc,1)
          end

       --   print('==> Take Average')

      output1=torch.sum(out_avg,1)
      output1=output1/30

      -------------------Normalise-----------------------------
         local x=torch.max(output1)

        output1=torch.div(output1,x)

       local norm=torch.norm(output1)
            output1=torch.div(output1,norm)

      mustache[t]=output1




          


k=k+1
t=t+1
end

torch.save('mustache',mustache)

vgg=torch.cat(vgg,mustache)
--print('IN_GEN DATA')
--print(gen[1][1],gen[2][1])
----------------------------------------test model----------------------------------------------
--for t=1,size do
 -- local prob =model_test:forward(gen[t])
 -- maxval,maxid = prob:max(1)
 -- print(maxid[1])
--end

----------------------------------End of blondhair.lua----------------------------------------------



---------------------model.lua---------------------------------------------------------
torch.setdefaulttensortype('torch.FloatTensor')



model= nn.Sequential()
view_states=10226*16
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(opt.states, opt.states))
     model:add(nn.View(opt.states,1))
      model:add(nn.TemporalConvolution(1, 32, 11))
      model:add(nn.ReLU())
      model:add(nn.TemporalMaxPooling(3,2))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
     model:add(nn.TemporalConvolution(32, 16,9))
    model:add(nn.ReLU())
     
      -- stage 3 : standard 2-layer neural network
     model:add(nn.View(view_states))
    model:add(nn.Dropout(0.5))
      model:add(nn.Linear(view_states, 4096))
      model:add(nn.ReLU())
     model:add(nn.Linear(4096, 5))
     model:add( nn.SoftMax())
 criterion = nn.ClassNLLCriterion()

local labelsShuffle = torch.randperm((#labelsAll)[1])
--print(labelsShuffle)


print '==> Create Train set'
trainData = {
data = torch.Tensor(trsize,opt.states),
labels = torch.Tensor(trsize),
size = function() return trsize end
}

print '==> Create Test set'
--create test set:
 testData = {
data = torch.Tensor(tesize,opt.states),
labels = torch.Tensor(tesize),
size = function() return tesize end
}

for i=1,trsize do
trainData.data[i] = vgg[labelsShuffle[i]]:clone()
trainData.labels[i] = labelsAll[labelsShuffle[i]]
--print(labelsShuffle[i])
end

for i=trsize+1,tesize+trsize do
testData.data[i-trsize] = vgg[labelsShuffle[i]]:clone()
testData.labels[i-trsize] =  labelsAll[labelsShuffle[i]]
--print(testData.labels[i-trsize])
end

print '==> preprocessing data: normalize each feature  globally'

mean = trainData.data[{ {},{} }]:mean()
   std = trainData.data[{ {},{} }]:std()
   trainData.data[{ {},{} }]:add(-mean)
   trainData.data[{ {},{} }]:div(std)

   print(mean,std)
  -- var=torch.Tensor{mean,std}
  -- torch.save('mean.txt',var)

testData.data[{ {},{} }]:add(-mean)
   testData.data[{ {},{} }]:div(std)
---------------------------------end of model.lua-------------------------------------


