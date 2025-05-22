# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Main method to train the model."""


#!/usr/bin/python

import argparse
import sys
import time
import datasets
import img_text_composition_models
# import numpy as np
# from tensorboardX import SummaryWriter
import test_retrieval
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm as tqdm

torch.set_num_threads(3)


def parse_opt():
  """Parses the input arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', type=str, default='')
  parser.add_argument('--comment', type=str, default='test_notebook')
  parser.add_argument('--dataset', type=str, default='css3d')
  parser.add_argument(
      '--dataset_path', type=str, default='../imgcomsearch/CSSDataset/output')
  parser.add_argument('--model', type=str, default='tirg')
  parser.add_argument('--embed_dim', type=int, default=512)
  parser.add_argument('--learning_rate', type=float, default=1e-2)
  parser.add_argument(
      '--learning_rate_decay_frequency', type=int, default=9999999)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--weight_decay', type=float, default=1e-6)
  parser.add_argument('--num_iters', type=int, default=210000)
  parser.add_argument('--loss', type=str, default='soft_triplet')
  parser.add_argument('--loader_num_workers', type=int, default=4)
  args = parser.parse_args()
  return args


def load_dataset(opt):
  """Loads the input datasets."""
  print('Reading dataset ', opt.dataset)
  if opt.dataset == 'css3d':
    trainset = datasets.CSSDataset(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.CSSDataset(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  elif opt.dataset == 'fashion200k':
    trainset = datasets.Fashion200k(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.Fashion200k(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  elif opt.dataset == 'mitstates':
    trainset = datasets.MITStates(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.MITStates(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  else:
    print('Invalid dataset', opt.dataset)
    sys.exit()

  print('trainset size:', len(trainset))
  print('testset size:', len(testset))
  return trainset, testset


def create_model_and_optimizer(opt, texts):
  """Builds the model and related optimizer."""
  print('Creating model and optimizer for', opt.model)
  if opt.model == 'imgonly':
    model = img_text_composition_models.SimpleModelImageOnly(
        texts, embed_dim=opt.embed_dim)
  elif opt.model == 'textonly':
    model = img_text_composition_models.SimpleModelTextOnly(
        texts, embed_dim=opt.embed_dim)
  elif opt.model == 'concat':
    model = img_text_composition_models.Concat(texts, embed_dim=opt.embed_dim)
  elif opt.model == 'tirg':
    model = img_text_composition_models.TIRG(texts, embed_dim=opt.embed_dim)
  elif opt.model == 'tirg_lastconv':
    model = img_text_composition_models.TIRGLastConv(
        texts, embed_dim=opt.embed_dim)
  else:
    print('Invalid model', opt.model)
    print('available: imgonly, textonly, concat, tirg or tirg_lastconv')
    sys.exit()
  # model = model.cuda()
  # model = model.to(torch.device('cpu'))
  # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  # model = model.to(device)
  print('---\n model:', model)
  # create optimizer
  params = []
  # low learning rate for pretrained layers on real image datasets
  if opt.dataset != 'css3d':
    params.append({
        'params': [p for p in model.img_model.fc.parameters()],
        'lr': opt.learning_rate
    })
    params.append({
        'params': [p for p in model.img_model.parameters()],
        'lr': 0.1 * opt.learning_rate
    })
  params.append({'params': [p for p in model.parameters()]})
  for _, p1 in enumerate(params):  # remove duplicated params
    for _, p2 in enumerate(params):
      if p1 is not p2:
        for p11 in p1['params']:
          for j, p22 in enumerate(p2['params']):
            if p11 is p22:
              p2['params'][j] = torch.tensor(0.0, requires_grad=True)
  optimizer = torch.optim.SGD(
      params, lr=opt.learning_rate, momentum=0.9, weight_decay=opt.weight_decay)
  return model, optimizer


def train_loop(opt, logger, trainset, testset, model, optimizer):
  """Function for train loop"""
  print('Begin training')
  losses_tracking = {}
  it = 0
  epoch = -1
  tic = time.time()
  while it < opt.num_iters:
    epoch += 1

    # show/log stats
    print('It', it, 'epoch', epoch, 'Elapsed time', round(time.time() - tic, 4), opt.comment)
    tic = time.time()
    for loss_name in losses_tracking:
      avg_loss = np.mean(losses_tracking[loss_name][-len(trainloader):])
      print('    Loss', loss_name, round(avg_loss, 4))
      logger.add_scalar(loss_name, avg_loss, it)
    logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], it)

    # test
    if epoch % 3 == 1:
      tests = []
      for name, dataset in [('train', trainset), ('test', testset)]:
        t = test_retrieval.test(opt, model, dataset)
        tests += [(name + ' ' + metric_name, metric_value)
                  for metric_name, metric_value in t]
      for metric_name, metric_value in tests:
        logger.add_scalar(metric_name, metric_value, it)
        print('    ', metric_name, round(metric_value, 4))

    # save checkpoint
    torch.save({
        'it': it,
        'opt': opt,
        'model_state_dict': model.state_dict(),
    },
               logger.file_writer.get_logdir() + '/latest_checkpoint.pth')

    # run trainning for 1 epoch
    model.train()
    trainloader = trainset.get_loader(
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=opt.loader_num_workers)


    def training_1_iter(data):
      assert type(data) is list
      img1 = np.stack([d['source_img_data'] for d in data])
      img1 = torch.from_numpy(img1).float()
      img1 = torch.autograd.Variable(img1).cuda()
      img2 = np.stack([d['target_img_data'] for d in data])
      img2 = torch.from_numpy(img2).float()
      img2 = torch.autograd.Variable(img2).cuda()
      mods = [str(d['mod']['str']) for d in data]

      # compute loss
      losses = []
      if opt.loss == 'soft_triplet':
        loss_value = model.compute_loss(
            img1, mods, img2, soft_triplet_loss=True)
      elif opt.loss == 'batch_based_classification':
        loss_value = model.compute_loss(
            img1, mods, img2, soft_triplet_loss=False)
      else:
        print('Invalid loss function', opt.loss)
        sys.exit()
      loss_name = opt.loss
      loss_weight = 1.0
      losses += [(loss_name, loss_weight, loss_value)]
      total_loss = sum([
          loss_weight * loss_value
          for loss_name, loss_weight, loss_value in losses
      ])
      assert not torch.isnan(total_loss)
      losses += [('total training loss', None, total_loss)]

      # track losses
      for loss_name, loss_weight, loss_value in losses:
        if loss_name not in losses_tracking:
          losses_tracking[loss_name] = []
        losses_tracking[loss_name].append(float(loss_value))

      # gradient descend
      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()

    for data in tqdm(trainloader, desc='Training for epoch ' + str(epoch)):
      it += 1
      training_1_iter(data)

      # decay learing rate
      if it >= opt.learning_rate_decay_frequency and it % opt.learning_rate_decay_frequency == 0:
        for g in optimizer.param_groups:
          g['lr'] *= 0.1

  print('Finished training')


def main():
  opt = parse_opt()
  print('Arguments:')
  for k in opt.__dict__.keys():
    print('    ', k, ':', str(opt.__dict__[k]))

  logger = SummaryWriter(comment=opt.comment)
  print('Log files saved to', logger.file_writer.get_logdir())
  for k in opt.__dict__.keys():
    logger.add_text(k, str(opt.__dict__[k]))

  trainset, testset = load_dataset(opt)
  model, optimizer = create_model_and_optimizer(opt, trainset.get_all_texts())

  train_loop(opt, logger, trainset, testset, model, optimizer)
  logger.close()


if __name__ == '__main__':
  # main()
  opt = parse_opt()
  print('Arguments:')
  for k in opt.__dict__.keys():
    print('    ', k, ':', str(opt.__dict__[k]))

  # logger = SummaryWriter(comment=opt.comment)
  # print('Log files saved to', logger.file_writer.get_logdir())
  # for k in opt.__dict__.keys():
  #   logger.add_text(k, str(opt.__dict__[k]))

  trainset, testset = load_dataset(opt)
  '''
  print('trainset.get_all_texts():',trainset.get_all_texts())
  ['黑色Perry弹力棉混纺绉布紧身裤', '黑色修身羊毛绉纱九分裤', '黑色修身羊毛绉纱九分裤']
  '''

  model, optimizer = create_model_and_optimizer(opt, trainset.get_all_texts())
  
  # Ensure input is on the same device as the model
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  '''1st try - start'''

  # # ### start the training
  # # train_loop(opt, logger, trainset, testset, model, optimizer)
  # # logger.close()

  '''load the pretrained model'''
  checkpoint = torch.load('Xueyan/checkpoint_fashion200k.pth',map_location=torch.device('cpu'),weights_only=False)
  # checkpoint = torch.load('Xueyan/checkpoint_fashion200k.pth',map_location=torch.device('cpu'))
  model.load_state_dict(checkpoint['model_state_dict'])

  # '''
  # # Define a sequence of transformations
  # transform = torchvision.transforms.Compose([
  #     transforms.Resize((256, 256)),  # Resize to 256x256
  #     transforms.ToTensor(),           # Convert image to PyTorch tensor (C, H, W) format
  #     transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values
  # ])
  # # Open an image and apply transformations
  # image = Image.open("example.jpg")
  # transformed_image = transform(image)

  # print(type(transformed_image))  # <class 'torch.Tensor'>
  # print(transformed_image.shape)  # (3, 256, 256) for RGB images
  # '''
  # ### test the pretrained model
  # # Define transformations
  # transform = trainset.transform

  # # Apply transformations
  # image = Image.open('./fashion200k/women/dresses/casual_and_day_dresses/51727804/51727804_0.jpeg')
  # input_tensor = transform(image)

  # # Add batch dimension (model expects [Batch, C, H, W])
  # input_batch = input_tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]
  # input_batch = input_batch.to(device)

  # # Get predictions
  # with torch.no_grad():  # Disable gradient calculations (faster inference)
  #     output = model(input_batch)

  # print('output.shape: ',output.shape)  # Output shape: [1, 1000] (1000 classes for ImageNet)

  '''1st try - end'''

  '''
  python main.py --dataset=fashion200k --dataset_path=../fashion200k \
  --num_iters=160000 --model=tirg --loss=batch_based_classification \
  --learning_rate_decay_frequency=50000 --comment=f200k_tirg
  '''

  '''2nd try - start'''
  # for name, dataset in [('train', trainset), ('test', testset)]:
  print('testset: ', testset)
  '''
  testset:  <datasets.Fashion200k object>
  img_path: where the pics are stored, ./
  split = 'train'/'test'
  transform = 
  '''
  t = test_retrieval.test(opt, model, testset)
  print('t: ', t)
  '''
  (after loading the pretrained model)t:  [
  ('recall_top1_correct_composition', 0.016995221027479093), 
  ('recall_top5_correct_composition', 0.06293309438470729), 
  ('recall_top10_correct_composition', 0.0966547192353644), 
  ('recall_top50_correct_composition', 0.2144862604540024), 
  ('recall_top100_correct_composition', 0.29277180406212666)
  ]
  '''

  '''
  (before loading the pretrained model)t:  [
  ('recall_top1_correct_composition', 0.0102), 
  ('recall_top5_correct_composition', 0.0289), 
  ('recall_top10_correct_composition', 0.0417), 
  ('recall_top50_correct_composition', 0.1011), 
  ('recall_top100_correct_composition', 0.1428)
  ]
  '''
  # print(name + ': ' + metric_name + '+' + round(metric_value, 4))
    
  '''2nd try - end'''

'''
Terminal output:
(base) menjiayu@MacBookAir tirg % python main.py --dataset=fashion200k --dataset_path=../fashion200k \
  --num_iters=160000 --model=tirg --loss=batch_based_classification \
  --learning_rate_decay_frequency=50000 --comment=f200k_tirg
Arguments:
     f : 
     comment : f200k_tirg
     dataset : fashion200k
     dataset_path : ./Fashion200k
     model : tirg
     embed_dim : 512
     learning_rate : 0.01
     learning_rate_decay_frequency : 50000
     batch_size : 32
     weight_decay : 1e-06
     num_iters : 160000
     loss : batch_based_classification
     loader_num_workers : 4
Reading dataset  fashion200k
read: pants_train_detect_all.txt
read: dress_train_detect_all.txt
read: jacket_train_detect_all.txt
read: skirt_train_detect_all.txt
read: top_train_detect_all.txt
Fashion200k: 172049 images
53099 unique cations
Modifiable images 106464
read: top_test_detect_all.txt
read: pants_test_detect_all.txt
read: dress_test_detect_all.txt
read: skirt_test_detect_all.txt
read: jacket_test_detect_all.txt
Fashion200k: 29789 images
trainset size: 172049
testset size: 29789
Creating model and optimizer for tirg
/opt/anaconda3/lib/python3.12/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/opt/anaconda3/lib/python3.12/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
testset:  <datasets.Fashion200k object at 0x13fa659a0>
length2:  10000
100%|█████████████████████████████████████████████████| 10000/10000 [00:21<00:00, 457.31it/s]
t:  [('recall_top1_correct_composition', 0.0102), ('recall_top5_correct_composition', 0.0289), ('recall_top10_correct_composition', 0.0417), ('recall_top50_correct_composition', 0.1011), ('recall_top100_correct_composition', 0.1428)]
'''

'''
% python main.py --dataset=fashion200k --dataset_path=../fashion200k \
  --num_iters=160000 --model=tirg --loss=batch_based_classification \
  --learning_rate_decay_frequency=50000 --comment=f200k_tirg
Arguments:
     f : 
     comment : f200k_tirg
     dataset : fashion200k
     dataset_path : ../fashion200k
     model : tirg
     embed_dim : 512
     learning_rate : 0.01
     learning_rate_decay_frequency : 50000
     batch_size : 32
     weight_decay : 1e-06
     num_iters : 160000
     loss : batch_based_classification
     loader_num_workers : 4
Reading dataset  fashion200k
read: dress_train_detect_all_cn.txt
read: skirt_train_detect_all_cn.txt
read: pants_train_detect_all_cn.txt
Fashion200k: 108714 images
30048 unique cations
Modifiable images 108714
read: top_test_detect_all_cn.txt
read: skirt_test_detect_all_cn.txt
read: jacket_test_detect_all_cn.txt
read: pants_test_detect_all_cn.txt
read: dress_test_detect_all_cn.txt
Fashion200k: 29789 images
trainset size: 108714
testset size: 29789
Creating model and optimizer for tirg
---
 model: TIRG(
  (normalization_layer): NormalizationLayer()
  (soft_triplet_loss): TripletLoss()
  (img_model): ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (layer1): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer2): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer3): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (avgpool): GlobalAvgPool2d()
    (fc): Sequential(
      (0): Linear(in_features=512, out_features=512, bias=True)
    )
  )
  (text_model): TextLSTMModel(
    (embedding_layer): Embedding(5590, 512)
    (lstm): LSTM(512, 512)
    (fc_output): Sequential(
      (0): Dropout(p=0.1, inplace=False)
      (1): Linear(in_features=512, out_features=512, bias=True)
    )
  )
  (gated_feature_composer): Sequential(
    (0): ConCatModule()
    (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=1024, out_features=512, bias=True)
  )
  (res_info_composer): Sequential(
    (0): ConCatModule()
    (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=1024, out_features=1024, bias=True)
    (4): ReLU()
    (5): Linear(in_features=1024, out_features=512, bias=True)
  )
)
testset:  <datasets.Fashion200k object at 0x7fbdaad19810>
len(test_queries):  33480
  0%|                                                                | 28/33480 [00:00<02:02, 272.84it/s]/opt/anaconda3/envs/cirr/lib/python3.7/site-packages/torch/nn/functional.py:1806: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
  warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
 39%|████████████████████████▎                                     | 13152/33480 [23:13<46:3 
 39%|████████████████████████▍                                     | 13184/33480 [23:17<45:3 
 39%|████████████████████████▍                                     | 13214/33480 [23:17<30:5 
 40%|████████████████████████▍                                     | 13226/33480 [23:21<42:1 
 40%|████████████████████████▌                                     | 13248/33480 [23:24<45:1 
 40%|████████████████████████▌                                     | 13280/33480 [23:28<42:0 
 40%|████████████████████████▋                                     | 13312/33480 [23:31<40:1 
 40%|████████████████████████▋                                     | 13344/33480 [23:35<38:5 
 40%|████████████████████████▊                                     | 13376/33480 [23:38<38:0 
 40%|████████████████████████▊                                     | 13408/33480 [23:42<37:2 
 40%|████████████████████████▉                                     | 13440/33480 [23:45<36:5 
 40%|████████████████████████▉                                     | 13472/33480 [23:49<36:3 
 40%|█████████████████████████                                     | 13504/33480 [23:53<37:4 
 40%|█████████████████████████                                     | 13536/33480 [23:56<38:0 
 41%|█████████████████████████▏                                    | 13568/33480 [24:00<38:3 
 41%|█████████████████████████▏                                    | 13597/33480 [24:00<27:5 
 41%|█████████████████████████▏                                    | 13607/33480 [24:04<40:4 
 41%|█████████████████████████▏                                    | 13632/33480 [24:08<43:0 
 41%|█████████████████████████▎                                    | 13664/33480 [24:12<42:3 
 41%|█████████████████████████▎                                    | 13696/33480 [24:16<41:0 
 41%|█████████████████████████▍                                    | 13728/33480 [24:19<39:3 
 41%|█████████████████████████▍                                    | 13760/33480 [24:23<38:4 
 41%|█████████████████████████▌                                    | 13792/33480 [24:26<38:0 
 41%|█████████████████████████▌                                    | 13824/33480 [24:30<38:0 
 41%|█████████████████████████▋                                    | 13856/33480 [24:34<38:0 
 41%|█████████████████████████▋                                    | 13888/33480 [24:38<37:4 
 42%|█████████████████████████▊                                    | 13920/33480 [24:42<38:3 
 42%|█████████████████████████▊                                    | 13952/33480 [24:45<38:1 
 42%|█████████████████████████▉                                    | 13984/33480 [24:49<37:1 
 42%|█████████████████████████▉                                    | 14016/33480 [24:52<37:0 
 42%|██████████████████████████                                    | 14043/33480 [24:52<27:2 
 42%|██████████████████████████                                    | 14052/33480 [24:56<38:3 
 42%|██████████████████████████                                    | 14080/33480 [24:59<39:2 
 42%|██████████████████████████▏                                   | 14112/33480 [25:03<38:0 
 42%|██████████████████████████▏                                   | 14144/33480 [25:07<37:1 
 42%|██████████████████████████▎                                   | 14176/33480 [25:10<36:4 
 42%|██████████████████████████▎                                   | 14208/33480 [25:14<36:1 
 43%|██████████████████████████▎                                   | 14240/33480 [25:17<36:3 
 43%|██████████████████████████▍                                   | 14264/33480 [25:18<27:4 
 43%|██████████████████████████▍                                   | 14272/33480 [25:21<39:4 
 43%|██████████████████████████▍                                   | 14304/33480 [25:25<38:3 
 43%|██████████████████████████▌                                   | 14336/33480 [25:29<38:2 
 43%|██████████████████████████▌                                   | 14368/33480 [25:32<37:4 
 43%|██████████████████████████▋                                   | 14400/33480 [25:36<36:5 
 43%|██████████████████████████▋                                   | 14432/33480 [25:39<36:3 
 43%|██████████████████████████▊                                   | 14464/33480 [25:43<35:5 
 43%|██████████████████████████▊                                   | 14496/33480 [25:46<35:2 
 43%|██████████████████████████▉                                   | 14528/33480 [25:50<35:0 
 43%|██████████████████████████▉                                   | 14560/33480 [25:53<34:4 
 44%|███████████████████████████                                   | 14592/33480 [25:57<34:3 
 44%|███████████████████████████                                   | 14624/33480 [26:00<34:2 
 44%|███████████████████████████▏                                  | 14656/33480 [26:04<34:0 
 44%|███████████████████████████▏                                  | 14688/33480 [26:07<34:0 
 44%|███████████████████████████▎                                  | 14720/33480 [26:11<34:0 
 44%|███████████████████████████▎                                  | 14752/33480 [26:14<33:5 
 44%|███████████████████████████▎                                  | 14774/33480 [26:14<26:2 
 44%|███████████████████████████▍                                  | 14784/33480 [26:18<36:5 
 44%|███████████████████████████▍                                  | 14816/33480 [26:21<35:4 
 44%|███████████████████████████▍                                  | 14848/33480 [26:25<34:5 
 44%|███████████████████████████▌                                  | 14880/33480 [26:28<34:2 
 45%|███████████████████████████▌                                  | 14912/33480 [26:31<33:5 
 45%|███████████████████████████▋                                  | 14944/33480 [26:35<33:4 
 45%|███████████████████████████▋                                  | 14976/33480 [26:38<33:2 
 45%|███████████████████████████▊                                  | 15008/33480 [26:42<33:1 
 45%|███████████████████████████▊                                  | 15040/33480 [26:45<33:1 
 45%|███████████████████████████▉                                  | 15072/33480 [26:49<33:1 
 45%|███████████████████████████▉                                  | 15104/33480 [26:52<33:0 
 45%|████████████████████████████                                  | 15136/33480 [26:56<34:0 
 45%|████████████████████████████                                  | 15168/33480 [27:00<34:2 
 45%|████████████████████████████▏                                 | 15200/33480 [27:03<33:5 
 45%|████████████████████████████▏                                 | 15232/33480 [27:07<33:4 
 46%|████████████████████████████▎                                 | 15264/33480 [27:10<33:3 
 46%|████████████████████████████▎                                 | 15296/33480 [27:14<33:1 
 46%|████████████████████████████▍                                 | 15328/33480 [27:17<32:5 
 46%|████████████████████████████▍                                 | 15360/33480 [27:21<32:5 
 46%|████████████████████████████▌                                 | 15392/33480 [27:24<32:4 
 46%|████████████████████████████▌                                 | 15424/33480 [27:28<32:5 
 46%|████████████████████████████▌                                 | 15456/33480 [27:31<33:0 
 46%|████████████████████████████▋                                 | 15488/33480 [27:35<32:5 
 46%|████████████████████████████▋                                 | 15520/33480 [27:38<32:4 
 46%|████████████████████████████▊                                 | 15552/33480 [27:42<33:2 
 47%|████████████████████████████▊                                 | 15584/33480 [27:45<33:1 
 47%|████████████████████████████▉                                 | 15616/33480 [27:49<32:5 
 47%|████████████████████████████▉                                 | 15648/33480 [27:52<32:4 
 47%|█████████████████████████████                                 | 15680/33480 [27:56<32:3 
 47%|█████████████████████████████                                 | 15712/33480 [27:59<32:2 
 47%|█████████████████████████████▏                                | 15744/33480 [28:03<32:0 
 47%|█████████████████████████████▏                                | 15776/33480 [28:06<32:0 
 47%|█████████████████████████████▎                                | 15808/33480 [28:10<31:5 
 47%|█████████████████████████████▎                                | 15840/33480 [28:13<31:4 
 47%|█████████████████████████████▍                                | 15872/33480 [28:17<31:4 
 48%|█████████████████████████████▍                                | 15904/33480 [28:20<31:3 
 48%|█████████████████████████████▌                                | 15936/33480 [28:23<31:2 
 48%|█████████████████████████████▌                                | 15968/33480 [28:27<31:2 
 48%|█████████████████████████████▋                                | 16000/33480 [28:30<31:1 
 48%|█████████████████████████████▋                                | 16032/33480 [28:34<31:2 
 48%|█████████████████████████████▋                                | 16064/33480 [28:37<31:2 
 48%|█████████████████████████████▊                                | 16096/33480 [28:41<31:1 
 48%|█████████████████████████████▊                                | 16128/33480 [28:44<31:2 
 48%|█████████████████████████████▉                                | 16160/33480 [28:48<31:1 
 48%|█████████████████████████████▉                                | 16192/33480 [28:51<31:2 
 48%|██████████████████████████████                                | 16224/33480 [28:55<31:4 
 49%|██████████████████████████████                                | 16256/33480 [28:58<31:3
 49%|██████████████████████████████▏                               | 16288/33480 [29:02<31:2 
 49%|██████████████████████████████▏                               | 16320/33480 [29:05<31:1 
 49%|██████████████████████████████▎                               | 16352/33480 [29:09<31:0 
 49%|██████████████████████████████▎                               | 16384/33480 [29:12<30:5 
 49%|██████████████████████████████▍                               | 16416/33480 [29:16<30:5 
 49%|██████████████████████████████▍                               | 16448/33480 [29:19<30:5 
 49%|██████████████████████████████▌                               | 16476/33480 [29:19<22:4 
 49%|██████████████████████████████▌                               | 16486/33480 [29:23<32:3 
 49%|██████████████████████████████▌                               | 16512/33480 [29:27<34:3 
 49%|██████████████████████████████▋                               | 16544/33480 [29:30<33:3 
 50%|██████████████████████████████▋                               | 16576/33480 [29:34<34:3 
 50%|██████████████████████████████▊                               | 16608/33480 [29:38<34:3 
 50%|██████████████████████████████▊                               | 16640/33480 [29:42<34:2 
 50%|██████████████████████████████▊                               | 16672/33480 [29:46<34:0 
 50%|██████████████████████████████▉                               | 16704/33480 [29:50<33:1 
 50%|██████████████████████████████▉                               | 16736/33480 [29:53<32:3 
 50%|███████████████████████████████                               | 16768/33480 [29:57<32:0 
 50%|███████████████████████████████                               | 16800/33480 [30:00<31:3 
 50%|███████████████████████████████▏                              | 16832/33480 [30:04<32:0 
 50%|███████████████████████████████▏                              | 16864/33480 [30:08<32:0 
 50%|███████████████████████████████▎                              | 16896/33480 [30:11<31:3 
 51%|███████████████████████████████▎                              | 16928/33480 [30:15<31:1 
 51%|███████████████████████████████▍                              | 16960/33480 [30:19<31:0 
 51%|███████████████████████████████▍                              | 16992/33480 [30:22<31:0 
 51%|███████████████████████████████▌                              | 17024/33480 [30:26<30:5 
 51%|███████████████████████████████▌                              | 17056/33480 [30:30<31:5 
 51%|███████████████████████████████▋                              | 17088/33480 [30:33<31:3 
 51%|███████████████████████████████▋                              | 17120/33480 [30:37<31:1 
 51%|███████████████████████████████▊                              | 17152/33480 [30:40<30:5 
 51%|███████████████████████████████▊                              | 17184/33480 [30:44<30:4 
 51%|███████████████████████████████▉                              | 17216/33480 [30:48<30:3 
 52%|███████████████████████████████▉                              | 17248/33480 [30:51<30:1 
 52%|████████████████████████████████                              | 17280/33480 [30:55<30:0 
 52%|████████████████████████████████                              | 17312/33480 [30:58<29:5 
 52%|████████████████████████████████                              | 17344/33480 [31:02<29:4 
 52%|████████████████████████████████▏                             | 17376/33480 [31:05<29:3 
 52%|████████████████████████████████▏                             | 17408/33480 [31:09<29:2 
 52%|████████████████████████████████▎                             | 17440/33480 [31:12<29:2 
 52%|████████████████████████████████▎                             | 17472/33480 [31:16<29:2 
 52%|████████████████████████████████▍                             | 17504/33480 [31:19<29:1 
 52%|████████████████████████████████▍                             | 17536/33480 [31:23<29:1 
 52%|████████████████████████████████▌                             | 17568/33480 [31:26<29:1 
 53%|████████████████████████████████▌                             | 17600/33480 [31:30<29:0 
 53%|████████████████████████████████▋                             | 17632/33480 [31:33<29:0 
 53%|████████████████████████████████▋                             | 17664/33480 [31:37<29:0 
 53%|████████████████████████████████▊                             | 17696/33480 [31:40<28:5 
 53%|████████████████████████████████▊                             | 17728/33480 [31:44<28:5 
 53%|████████████████████████████████▉                             | 17760/33480 [31:48<28:4 
 53%|████████████████████████████████▉                             | 17792/33480 [31:51<28:4 
 53%|█████████████████████████████████                             | 17824/33480 [31:55<28:4 
 53%|█████████████████████████████████                             | 17856/33480 [31:58<28:4 
 53%|█████████████████████████████████▏                            | 17888/33480 [32:02<28:4 
 54%|█████████████████████████████████▏                            | 17920/33480 [32:05<28:4 
 54%|█████████████████████████████████▏                            | 17952/33480 [32:09<28:3 
 54%|█████████████████████████████████▎                            | 17984/33480 [32:12<28:2 
 54%|█████████████████████████████████▎                            | 18016/33480 [32:16<28:1 
 54%|█████████████████████████████████▍                            | 18048/33480 [32:19<28:1 
 54%|█████████████████████████████████▍                            | 18080/33480 [32:23<27:5 
 54%|█████████████████████████████████▌                            | 18112/33480 [32:26<28:1 
 54%|█████████████████████████████████▌                            | 18144/33480 [32:30<28:0 
 54%|█████████████████████████████████▋                            | 18176/33480 [32:33<27:5 
 54%|█████████████████████████████████▋                            | 18208/33480 [32:37<28:5 
 54%|█████████████████████████████████▊                            | 18240/33480 [32:41<29:0 
 55%|█████████████████████████████████▊                            | 18272/33480 [32:44<28:2 
 55%|█████████████████████████████████▉                            | 18304/33480 [32:48<28:0 
 55%|█████████████████████████████████▉                            | 18336/33480 [32:51<28:0 
 55%|██████████████████████████████████                            | 18368/33480 [32:55<28:1 
 55%|██████████████████████████████████                            | 18400/33480 [32:59<28:3 
 55%|██████████████████████████████████▏                           | 18432/33480 [33:02<28:0 
 55%|██████████████████████████████████▏                           | 18464/33480 [33:06<28:0 
 55%|██████████████████████████████████▎                           | 18496/33480 [33:09<27:3 
 55%|██████████████████████████████████▎                           | 18528/33480 [33:13<27:1 
 55%|██████████████████████████████████▎                           | 18560/33480 [33:16<27:1 
 56%|██████████████████████████████████▍                           | 18592/33480 [33:20<27:4 
 56%|██████████████████████████████████▍                           | 18624/33480 [33:23<27:3 
 56%|██████████████████████████████████▌                           | 18656/33480 [33:27<27:4 
 56%|██████████████████████████████████▌                           | 18688/33480 [33:31<27:4 
 56%|██████████████████████████████████▋                           | 18720/33480 [33:34<27:3 
 56%|██████████████████████████████████▋                           | 18752/33480 [33:38<27:4 
 56%|██████████████████████████████████▊                           | 18784/33480 [33:42<28:2 
 56%|██████████████████████████████████▊                           | 18816/33480 [33:46<28:2 
 56%|██████████████████████████████████▉                           | 18848/33480 [33:49<27:5 
 56%|██████████████████████████████████▉                           | 18880/33480 [33:53<27:3 
 56%|███████████████████████████████████                           | 18912/33480 [33:56<27:5 
 57%|███████████████████████████████████                           | 18944/33480 [34:00<27:3 
 57%|███████████████████████████████████▏                          | 18976/33480 [34:04<28:0 
 57%|███████████████████████████████████▏                          | 19008/33480 [34:08<27:5 
 57%|███████████████████████████████████▎                          | 19040/33480 [34:11<27:5 
 57%|███████████████████████████████████▎                          | 19072/33480 [34:16<28:5 
 57%|███████████████████████████████████▍                          | 19104/33480 [34:20<29:3 
 57%|███████████████████████████████████▍                          | 19136/33480 [34:23<29:0 
 57%|███████████████████████████████████▍                          | 19168/33480 [34:27<28:2 
 57%|███████████████████████████████████▌                          | 19200/33480 [34:31<29:1 
 57%|███████████████████████████████████▌                          | 19232/33480 [34:35<29:0 
 58%|███████████████████████████████████▋                          | 19264/33480 [34:39<29:2 
 58%|███████████████████████████████████▋                          | 19295/33480 [34:39<20:5 
 58%|███████████████████████████████████▊                          | 19305/33480 [34:43<29:3 
 58%|███████████████████████████████████▊                          | 19328/33480 [34:47<32:0 
 58%|███████████████████████████████████▊                          | 19360/33480 [34:51<30:3 
 58%|███████████████████████████████████▉                          | 19392/33480 [34:55<29:3 
 58%|███████████████████████████████████▉                          | 19424/33480 [34:58<28:3 
 58%|████████████████████████████████████                          | 19456/33480 [35:02<28:1 
 58%|████████████████████████████████████                          | 19488/33480 [35:06<27:4 
 58%|████████████████████████████████████▏                         | 19520/33480 [35:10<27:2 
 58%|████████████████████████████████████▏                         | 19552/33480 [35:13<26:5 
 58%|████████████████████████████████████▎                         | 19584/33480 [35:17<26:2 
 59%|████████████████████████████████████▎                         | 19616/33480 [35:20<26:1 
 59%|████████████████████████████████████▍                         | 19648/33480 [35:24<26:1 
 59%|████████████████████████████████████▍                         | 19680/33480 [35:28<27:1 
 59%|████████████████████████████████████▌                         | 19712/33480 [35:32<27:3 
 59%|████████████████████████████████████▌                         | 19744/33480 [35:36<27:0 
 59%|████████████████████████████████████▌                         | 19776/33480 [35:39<26:4 
 59%|████████████████████████████████████▋                         | 19806/33480 [35:39<19:1 
 59%|████████████████████████████████████▋                         | 19816/33480 [35:43<26:5 
 59%|████████████████████████████████████▋                         | 19840/33480 [35:47<29:0 
 59%|████████████████████████████████████▊                         | 19872/33480 [35:50<27:5 
 59%|████████████████████████████████████▊                         | 19904/33480 [35:54<27:0 
 60%|████████████████████████████████████▉                         | 19936/33480 [35:58<26:4 
 60%|████████████████████████████████████▉                         | 19968/33480 [36:01<26:1 
 60%|█████████████████████████████████████                         | 20000/33480 [36:05<26:3 
 60%|█████████████████████████████████████                         | 20032/33480 [36:09<26:5 
 60%|█████████████████████████████████████▏                        | 20064/33480 [36:13<26:2 
 60%|█████████████████████████████████████▏                        | 20096/33480 [36:17<26:3 
 60%|█████████████████████████████████████▎                        | 20128/33480 [36:20<26:2 
 60%|█████████████████████████████████████▎                        | 20160/33480 [36:24<25:4 
 60%|█████████████████████████████████████▍                        | 20192/33480 [36:27<25:2 
 60%|█████████████████████████████████████▍                        | 20224/33480 [36:31<25:3 
 61%|█████████████████████████████████████▌                        | 20256/33480 [36:35<25:2 
 61%|█████████████████████████████████████▌                        | 20288/33480 [36:39<25:2 
 61%|█████████████████████████████████████▋                        | 20320/33480 [36:42<25:1 
 61%|█████████████████████████████████████▋                        | 20352/33480 [36:46<24:5 
 61%|█████████████████████████████████████▋                        | 20384/33480 [36:49<24:4 
 61%|█████████████████████████████████████▊                        | 20416/33480 [36:53<24:4 
 61%|█████████████████████████████████████▊                        | 20448/33480 [36:57<24:5 
 61%|█████████████████████████████████████▉                        | 20476/33480 [36:57<18:1 
 61%|█████████████████████████████████████▉                        | 20485/33480 [37:01<26:0 
 61%|█████████████████████████████████████▉                        | 20512/33480 [37:04<26:4 
 61%|██████████████████████████████████████                        | 20544/33480 [37:08<25:3 
 61%|██████████████████████████████████████                        | 20576/33480 [37:11<24:5 
 62%|██████████████████████████████████████▏                       | 20608/33480 [37:15<24:3 
 62%|██████████████████████████████████████▏                       | 20640/33480 [37:18<24:1 
 62%|██████████████████████████████████████▎                       | 20672/33480 [37:22<24:0 
 62%|██████████████████████████████████████▎                       | 20704/33480 [37:25<23:5 
 62%|██████████████████████████████████████▍                       | 20736/33480 [37:29<23:4 
 62%|██████████████████████████████████████▍                       | 20768/33480 [37:33<23:5 
 62%|██████████████████████████████████████▍                       | 20771/33480 [37:33<23:1 
 62%|██████████████████████████████████████▍                       | 20785/33480 [37:33<18:5 
 62%|██████████████████████████████████████▌                       | 20800/33480 [37:37<29:4 
 62%|██████████████████████████████████████▌                       | 20830/33480 [37:38<17:5 
 62%|██████████████████████████████████████▌                       | 20842/33480 [37:42<27:5 
 62%|██████████████████████████████████████▋                       | 20864/33480 [37:45<30:0 
 62%|██████████████████████████████████████▋                       | 20896/33480 [37:49<27:2 
 63%|██████████████████████████████████████▊                       | 20928/33480 [37:53<26:3 
 63%|██████████████████████████████████████▊                       | 20960/33480 [37:56<25:2 
 63%|██████████████████████████████████████▊                       | 20992/33480 [38:00<25:1 
 63%|██████████████████████████████████████▉                       | 21023/33480 [38:00<17:2 
 63%|██████████████████████████████████████▉                       | 21034/33480 [38:04<24:2 
 63%|██████████████████████████████████████▉                       | 21056/33480 [38:07<27:1 
 63%|███████████████████████████████████████                       | 21088/33480 [38:11<25:5 
 63%|███████████████████████████████████████                       | 21119/33480 [38:11<17:2 
 63%|███████████████████████████████████████▏                      | 21132/33480 [38:15<23:5 
 63%|███████████████████████████████████████▏                      | 21152/33480 [38:19<29:2 
 63%|███████████████████████████████████████▏                      | 21180/33480 [38:19<19:2 
 63%|███████████████████████████████████████▏                      | 21193/33480 [38:23<27:4 
 63%|███████████████████████████████████████▎                      | 21216/33480 [38:27<29:5 
 63%|███████████████████████████████████████▎                      | 21247/33480 [38:27<18:4 
 64%|███████████████████████████████████████▎                      | 21261/33480 [38:31<25:2 
 64%|███████████████████████████████████████▍                      | 21280/33480 [38:35<29:5 
 64%|███████████████████████████████████████▍                      | 21312/33480 [38:39<27:1 
 64%|███████████████████████████████████████▌                      | 21344/33480 [38:42<26:0 
 64%|███████████████████████████████████████▌                      | 21376/33480 [38:46<25:0 
 64%|███████████████████████████████████████▋                      | 21408/33480 [38:50<24:0 
 64%|███████████████████████████████████████▋                      | 21440/33480 [38:53<23:3 
 64%|███████████████████████████████████████▊                      | 21472/33480 [38:57<22:5 
 64%|███████████████████████████████████████▊                      | 21504/33480 [39:00<22:4 
 64%|███████████████████████████████████████▉                      | 21536/33480 [39:04<22:2 
 64%|███████████████████████████████████████▉                      | 21568/33480 [39:07<22:1 
 65%|████████████████████████████████████████                      | 21600/33480 [39:11<21:5 
 65%|████████████████████████████████████████                      | 21632/33480 [39:14<21:4 
 65%|████████████████████████████████████████                      | 21664/33480 [39:18<21:3 
 65%|████████████████████████████████████████▏                     | 21696/33480 [39:21<21:4 
 65%|████████████████████████████████████████▏                     | 21728/33480 [39:25<21:5 
 65%|████████████████████████████████████████▎                     | 21752/33480 [39:25<16:4 
 65%|████████████████████████████████████████▎                     | 21760/33480 [39:29<23:5 
 65%|████████████████████████████████████████▎                     | 21792/33480 [39:33<24:2 
 65%|████████████████████████████████████████▍                     | 21824/33480 [39:37<25:0 
 65%|████████████████████████████████████████▍                     | 21856/33480 [39:41<24:1 
 65%|████████████████████████████████████████▌                     | 21888/33480 [39:45<23:2 
 65%|████████████████████████████████████████▌                     | 21920/33480 [39:48<22:5 
 66%|████████████████████████████████████████▋                     | 21952/33480 [39:52<22:2 
 66%|████████████████████████████████████████▋                     | 21984/33480 [39:55<22:0 
 66%|████████████████████████████████████████▊                     | 22016/33480 [39:59<22:1 
 66%|████████████████████████████████████████▊                     | 22048/33480 [40:03<22:2 
 66%|████████████████████████████████████████▉                     | 22080/33480 [40:07<21:5 
 66%|████████████████████████████████████████▉                     | 22112/33480 [40:10<21:4 
 66%|█████████████████████████████████████████                     | 22144/33480 [40:14<21:2 
 66%|█████████████████████████████████████████                     | 22176/33480 [40:17<21:1 
 66%|█████████████████████████████████████████▏                    | 22208/33480 [40:21<21:0 
 66%|█████████████████████████████████████████▏                    | 22240/33480 [40:24<20:5 
 67%|█████████████████████████████████████████▏                    | 22272/33480 [40:28<21:2 
 67%|█████████████████████████████████████████▎                    | 22304/33480 [40:32<21:1 
 67%|█████████████████████████████████████████▎                    | 22336/33480 [40:35<21:0 
 67%|█████████████████████████████████████████▍                    | 22368/33480 [40:39<20:5 
 67%|█████████████████████████████████████████▍                    | 22400/33480 [40:42<20:3 
 67%|█████████████████████████████████████████▌                    | 22432/33480 [40:46<20:2 
 67%|█████████████████████████████████████████▌                    | 22464/33480 [40:49<20:1 
 67%|█████████████████████████████████████████▋                    | 22496/33480 [40:53<20:0 
 67%|█████████████████████████████████████████▋                    | 22528/33480 [40:57<20:1 
 67%|█████████████████████████████████████████▊                    | 22558/33480 [40:57<14:3 
 67%|█████████████████████████████████████████▊                    | 22568/33480 [41:00<20:5 
 67%|█████████████████████████████████████████▊                    | 22592/33480 [41:04<22:3 
 68%|█████████████████████████████████████████▉                    | 22624/33480 [41:07<21:4 
 68%|█████████████████████████████████████████▉                    | 22656/33480 [41:11<21:0 
 68%|██████████████████████████████████████████                    | 22688/33480 [41:15<20:3 
 68%|██████████████████████████████████████████                    | 22720/33480 [41:18<20:1 
 68%|██████████████████████████████████████████▏                   | 22752/33480 [41:22<20:0 
 68%|██████████████████████████████████████████▏                   | 22784/33480 [41:25<19:5 
 68%|██████████████████████████████████████████▎                   | 22816/33480 [41:29<19:4 
 68%|██████████████████████████████████████████▎                   | 22848/33480 [41:32<19:3 
 68%|██████████████████████████████████████████▎                   | 22880/33480 [41:36<19:2 
 68%|██████████████████████████████████████████▍                   | 22912/33480 [41:39<19:1 
 69%|██████████████████████████████████████████▍                   | 22944/33480 [41:42<19:1 
 69%|██████████████████████████████████████████▌                   | 22976/33480 [41:46<19:0 
 69%|██████████████████████████████████████████▌                   | 23008/33480 [41:49<19:0 
 69%|██████████████████████████████████████████▋                   | 23040/33480 [41:53<18:5 
 69%|██████████████████████████████████████████▋                   | 23072/33480 [41:57<19:0 
 69%|██████████████████████████████████████████▊                   | 23104/33480 [42:00<19:0 
 69%|██████████████████████████████████████████▊                   | 23124/33480 [42:00<15:1 
 69%|██████████████████████████████████████████▊                   | 23136/33480 [42:04<20:2 
 69%|██████████████████████████████████████████▉                   | 23168/33480 [42:07<19:5 
 69%|██████████████████████████████████████████▉                   | 23200/33480 [42:11<19:2 
 69%|███████████████████████████████████████████                   | 23232/33480 [42:14<19:0 
 69%|███████████████████████████████████████████                   | 23264/33480 [42:18<18:5 
 70%|███████████████████████████████████████████▏                  | 23296/33480 [42:21<18:4 
 70%|███████████████████████████████████████████▏                  | 23328/33480 [42:25<18:3 
 70%|███████████████████████████████████████████▎                  | 23360/33480 [42:28<18:2 
 70%|███████████████████████████████████████████▎                  | 23392/33480 [42:32<18:1 
 70%|███████████████████████████████████████████▍                  | 23424/33480 [42:35<18:3 
 70%|███████████████████████████████████████████▍                  | 23456/33480 [42:39<19:0 
 70%|███████████████████████████████████████████▍                  | 23485/33480 [42:39<13:4 
 70%|███████████████████████████████████████████▌                  | 23495/33480 [42:43<19:4 
 70%|███████████████████████████████████████████▌                  | 23520/33480 [42:47<21:2 
 70%|███████████████████████████████████████████▌                  | 23552/33480 [42:50<20:3 
 70%|███████████████████████████████████████████▋                  | 23584/33480 [42:54<19:4 
 71%|███████████████████████████████████████████▋                  | 23616/33480 [42:58<19:0 
 71%|███████████████████████████████████████████▊                  | 23648/33480 [43:01<18:5 
 71%|███████████████████████████████████████████▊                  | 23680/33480 [43:05<18:3 
 71%|███████████████████████████████████████████▉                  | 23712/33480 [43:08<18:1 
 71%|███████████████████████████████████████████▉                  | 23744/33480 [43:12<18:0 
 71%|████████████████████████████████████████████                  | 23776/33480 [43:15<17:5 
 71%|████████████████████████████████████████████                  | 23808/33480 [43:19<17:4 
 71%|████████████████████████████████████████████▏                 | 23840/33480 [43:22<17:4 
 71%|████████████████████████████████████████████▏                 | 23872/33480 [43:26<18:0 
 71%|████████████████████████████████████████████▎                 | 23904/33480 [43:30<18:2 
 71%|████████████████████████████████████████████▎                 | 23936/33480 [43:33<18:0 
 72%|████████████████████████████████████████████▍                 | 23968/33480 [43:37<17:5 
 72%|████████████████████████████████████████████▍                 | 24000/33480 [43:40<17:3 
 72%|████████████████████████████████████████████▌                 | 24032/33480 [43:44<17:3 
 72%|████████████████████████████████████████████▌                 | 24064/33480 [43:47<17:1 
 72%|████████████████████████████████████████████▌                 | 24093/33480 [43:47<12:3 
 72%|████████████████████████████████████████████▋                 | 24103/33480 [43:51<17:3 
 72%|████████████████████████████████████████████▋                 | 24128/33480 [43:54<18:4 
 72%|████████████████████████████████████████████▋                 | 24160/33480 [43:58<18:0 
 72%|████████████████████████████████████████████▊                 | 24192/33480 [44:01<17:3 
 72%|████████████████████████████████████████████▊                 | 24224/33480 [44:05<17:1 
 72%|████████████████████████████████████████████▉                 | 24256/33480 [44:08<16:5 
 73%|████████████████████████████████████████████▉                 | 24288/33480 [44:12<16:5 
 73%|█████████████████████████████████████████████                 | 24320/33480 [44:15<16:5 
 73%|█████████████████████████████████████████████                 | 24351/33480 [44:15<11:5 
 73%|█████████████████████████████████████████████                 | 24362/33480 [44:20<17:4 
 73%|█████████████████████████████████████████████▏                | 24384/33480 [44:23<19:5 
 73%|█████████████████████████████████████████████▏                | 24416/33480 [44:27<18:3 
 73%|█████████████████████████████████████████████▎                | 24448/33480 [44:30<17:5 
 73%|█████████████████████████████████████████████▎                | 24480/33480 [44:34<17:3 
 73%|█████████████████████████████████████████████▍                | 24512/33480 [44:37<17:1 
 73%|█████████████████████████████████████████████▍                | 24544/33480 [44:41<17:0 
 73%|█████████████████████████████████████████████▌                | 24576/33480 [44:45<16:5 
 74%|█████████████████████████████████████████████▌                | 24608/33480 [44:48<16:3 
 74%|█████████████████████████████████████████████▋                | 24640/33480 [44:52<16:2 
 74%|█████████████████████████████████████████████▋                | 24672/33480 [44:55<16:2 
 74%|█████████████████████████████████████████████▋                | 24704/33480 [44:59<16:1 
 74%|█████████████████████████████████████████████▊                | 24736/33480 [45:02<16:0 
 74%|█████████████████████████████████████████████▊                | 24768/33480 [45:06<15:5 
 74%|█████████████████████████████████████████████▉                | 24800/33480 [45:09<15:4 
 74%|█████████████████████████████████████████████▉                | 24832/33480 [45:13<15:4
 74%|██████████████████████████████████████████████                | 24856/33480 [45:13<11:5 
 74%|██████████████████████████████████████████████                | 24864/33480 [45:16<17:0 
 74%|██████████████████████████████████████████████                | 24896/33480 [45:20<16:2 
 74%|██████████████████████████████████████████████▏               | 24928/33480 [45:23<16:0 
 75%|██████████████████████████████████████████████▏               | 24960/33480 [45:27<15:5 
 75%|██████████████████████████████████████████████▎               | 24992/33480 [45:30<15:4 
75%|██████████████████████████████████████████████▎               | 25024/33480 [45:33<15:2 
75%|██████████████████████████████████████████████▍               | 25056/33480 [45:37<15:1 
75%|██████████████████████████████████████████████▍               | 25088/33480 [45:40<15:1 
75%|██████████████████████████████████████████████▌               | 25120/33480 [45:44<15:4 
75%|██████████████████████████████████████████████▌               | 25152/33480 [45:48<15:5 
75%|██████████████████████████████████████████████▋               | 25184/33480 [45:52<15:4 
75%|██████████████████████████████████████████████▋               | 25216/33480 [45:55<15:4 
75%|██████████████████████████████████████████████▊               | 25248/33480 [45:59<15:3 
75%|██████████████████████████████████████████████▊               | 25264/33480 [45:59<12:5 
76%|██████████████████████████████████████████████▊               | 25280/33480 [46:02<16:1 
76%|██████████████████████████████████████████████▊               | 25312/33480 [46:06<15:4 
76%|██████████████████████████████████████████████▉               | 25344/33480 [46:10<15:2 
76%|██████████████████████████████████████████████▉               | 25376/33480 [46:13<15:0 
76%|███████████████████████████████████████████████               | 25408/33480 [46:16<14:5 
76%|███████████████████████████████████████████████               | 25440/33480 [46:20<14:4 
76%|███████████████████████████████████████████████▏              | 25472/33480 [46:23<14:3 
76%|███████████████████████████████████████████████▏              | 25504/33480 [46:27<14:3 
76%|███████████████████████████████████████████████▎              | 25536/33480 [46:30<14:2 
76%|███████████████████████████████████████████████▎              | 25568/33480 [46:34<14:2 
76%|███████████████████████████████████████████████▍              | 25600/33480 [46:37<14:1 
77%|███████████████████████████████████████████████▍              | 25632/33480 [46:41<14:1 
77%|███████████████████████████████████████████████▌              | 25664/33480 [46:44<14:0 
77%|███████████████████████████████████████████████▌              | 25696/33480 [46:48<14:0 
77%|███████████████████████████████████████████████▋              | 25728/33480 [46:51<13:5 
77%|███████████████████████████████████████████████▋              | 25760/33480 [46:55<13:5 
77%|███████████████████████████████████████████████▊              | 25792/33480 [46:58<13:4 
77%|███████████████████████████████████████████████▊              | 25824/33480 [47:01<13:4 
77%|███████████████████████████████████████████████▉              | 25856/33480 [47:05<13:4 
77%|███████████████████████████████████████████████▉              | 25885/33480 [47:05<09:5 
77%|███████████████████████████████████████████████▉              | 25895/33480 [47:08<13:5 
77%|████████████████████████████████████████████████              | 25920/33480 [47:12<14:5 
78%|████████████████████████████████████████████████              | 25952/33480 [47:15<14:2 
78%|████████████████████████████████████████████████              | 25984/33480 [47:19<14:0 
78%|████████████████████████████████████████████████▏             | 26016/33480 [47:22<13:4 
78%|████████████████████████████████████████████████▏             | 26048/33480 [47:26<13:3 
78%|████████████████████████████████████████████████▎             | 26080/33480 [47:29<13:3 
78%|████████████████████████████████████████████████▎             | 26112/33480 [47:33<13:2 
78%|████████████████████████████████████████████████▍             | 26144/33480 [47:36<13:1 
78%|████████████████████████████████████████████████▍             | 26176/33480 [47:39<13:1 
78%|████████████████████████████████████████████████▌             | 26208/33480 [47:43<13:2 
78%|████████████████████████████████████████████████▌             | 26240/33480 [47:47<13:1 
78%|████████████████████████████████████████████████▋             | 26272/33480 [47:50<13:0 
79%|████████████████████████████████████████████████▋             | 26304/33480 [47:54<13:0 
79%|████████████████████████████████████████████████▊             | 26336/33480 [47:57<13:0 
79%|████████████████████████████████████████████████▊             | 26368/33480 [48:01<12:5 
79%|████████████████████████████████████████████████▉             | 26400/33480 [48:04<12:5 
79%|████████████████████████████████████████████████▉             | 26432/33480 [48:08<13:1 
79%|█████████████████████████████████████████████████             | 26464/33480 [48:12<13:1 
79%|█████████████████████████████████████████████████             | 26496/33480 [48:15<13:1 
79%|█████████████████████████████████████████████████▏            | 26528/33480 [48:19<13:0 
79%|█████████████████████████████████████████████████▏            | 26560/33480 [48:22<12:5 
79%|█████████████████████████████████████████████████▏            | 26592/33480 [48:26<12:4 
80%|█████████████████████████████████████████████████▎            | 26624/33480 [48:29<12:4 
80%|█████████████████████████████████████████████████▎            | 26656/33480 [48:33<12:3 
80%|█████████████████████████████████████████████████▍            | 26688/33480 [48:36<12:2 
80%|█████████████████████████████████████████████████▍            | 26720/33480 [48:40<12:2 
80%|█████████████████████████████████████████████████▌            | 26752/33480 [48:43<12:2 
80%|█████████████████████████████████████████████████▌            | 26784/33480 [48:47<12:1 
80%|█████████████████████████████████████████████████▋            | 26816/33480 [48:50<12:1 
80%|█████████████████████████████████████████████████▋            | 26848/33480 [48:54<12:1 
80%|█████████████████████████████████████████████████▊            | 26880/33480 [48:58<12:0 
80%|█████████████████████████████████████████████████▊            | 26912/33480 [49:01<12:2 
80%|█████████████████████████████████████████████████▉            | 26944/33480 [49:05<12:4 
81%|█████████████████████████████████████████████████▉            | 26976/33480 [49:09<12:5 
81%|██████████████████████████████████████████████████            | 27008/33480 [49:13<12:3 
81%|██████████████████████████████████████████████████            | 27040/33480 [49:16<12:2 
81%|██████████████████████████████████████████████████▏           | 27072/33480 [49:20<12:0 
81%|██████████████████████████████████████████████████▏           | 27104/33480 [49:23<11:5 
81%|██████████████████████████████████████████████████▎           | 27136/33480 [49:27<11:5 
81%|██████████████████████████████████████████████████▎           | 27168/33480 [49:31<11:4 
81%|██████████████████████████████████████████████████▎           | 27200/33480 [49:34<11:3 
81%|██████████████████████████████████████████████████▍           | 27232/33480 [49:38<11:3 
81%|██████████████████████████████████████████████████▍           | 27264/33480 [49:41<11:2 
82%|██████████████████████████████████████████████████▌           | 27296/33480 [49:45<11:2 
82%|██████████████████████████████████████████████████▌           | 27328/33480 [49:48<11:1 
82%|██████████████████████████████████████████████████▋           | 27360/33480 [49:52<11:1 
82%|██████████████████████████████████████████████████▋           | 27392/33480 [49:55<11:0 
82%|██████████████████████████████████████████████████▊           | 27424/33480 [49:59<11:0 
82%|██████████████████████████████████████████████████▊           | 27456/33480 [50:02<11:0 
82%|██████████████████████████████████████████████████▉           | 27488/33480 [50:06<11:0 
82%|██████████████████████████████████████████████████▉           | 27520/33480 [50:09<11:0 
82%|███████████████████████████████████████████████████           | 27552/33480 [50:13<10:5 
82%|███████████████████████████████████████████████████           | 27584/33480 [50:16<10:4 
82%|███████████████████████████████████████████████████▏          | 27616/33480 [50:20<10:4 
83%|███████████████████████████████████████████████████▏          | 27648/33480 [50:23<10:3 
83%|███████████████████████████████████████████████████▎          | 27680/33480 [50:27<10:3 
83%|███████████████████████████████████████████████████▎          | 27711/33480 [50:27<07:3 
83%|███████████████████████████████████████████████████▎          | 27721/33480 [50:30<10:3 
83%|███████████████████████████████████████████████████▍          | 27744/33480 [50:34<11:3 
83%|███████████████████████████████████████████████████▍          | 27776/33480 [50:37<11:0 
83%|███████████████████████████████████████████████████▍          | 27808/33480 [50:41<10:4 
83%|███████████████████████████████████████████████████▌          | 27840/33480 [50:44<10:2 
83%|███████████████████████████████████████████████████▌          | 27872/33480 [50:48<10:1 
83%|███████████████████████████████████████████████████▋          | 27904/33480 [50:51<10:0 
83%|███████████████████████████████████████████████████▋          | 27936/33480 [50:55<10:0 
84%|███████████████████████████████████████████████████▊          | 27968/33480 [50:58<09:5 
84%|███████████████████████████████████████████████████▊          | 28000/33480 [51:01<09:5 
84%|███████████████████████████████████████████████████▉          | 28032/33480 [51:05<09:4 
84%|███████████████████████████████████████████████████▉          | 28064/33480 [51:08<09:4 
84%|████████████████████████████████████████████████████          | 28096/33480 [51:12<09:3 
84%|████████████████████████████████████████████████████          | 28128/33480 [51:15<09:4 
84%|████████████████████████████████████████████████████▏         | 28160/33480 [51:19<10:0 
84%|████████████████████████████████████████████████████▏         | 28192/33480 [51:23<10:0 
84%|████████████████████████████████████████████████████▎         | 28224/33480 [51:26<09:5 
84%|████████████████████████████████████████████████████▎         | 28256/33480 [51:30<09:4 
84%|████████████████████████████████████████████████████▍         | 28288/33480 [51:33<09:3 
85%|████████████████████████████████████████████████████▍         | 28320/33480 [51:37<09:2 
85%|████████████████████████████████████████████████████▌         | 28352/33480 [51:40<09:2 
85%|████████████████████████████████████████████████████▌         | 28384/33480 [51:44<09:1 
85%|████████████████████████████████████████████████████▌         | 28416/33480 [51:47<09:1 
85%|████████████████████████████████████████████████████▋         | 28448/33480 [51:51<09:0 
85%|████████████████████████████████████████████████████▋         | 28480/33480 [51:54<08:5 
85%|████████████████████████████████████████████████████▊         | 28512/33480 [51:58<08:5 
85%|████████████████████████████████████████████████████▊         | 28544/33480 [52:01<08:5 
85%|████████████████████████████████████████████████████▉         | 28576/33480 [52:04<08:4 
85%|████████████████████████████████████████████████████▉         | 28608/33480 [52:08<08:4 
86%|█████████████████████████████████████████████████████         | 28640/33480 [52:11<08:3 
86%|█████████████████████████████████████████████████████         | 28672/33480 [52:15<08:3 
86%|█████████████████████████████████████████████████████▏        | 28704/33480 [52:18<08:3 
86%|█████████████████████████████████████████████████████▏        | 28736/33480 [52:22<08:3 
86%|█████████████████████████████████████████████████████▎        | 28768/33480 [52:25<08:3 
86%|█████████████████████████████████████████████████████▎        | 28800/33480 [52:29<08:4 
86%|█████████████████████████████████████████████████████▍        | 28832/33480 [52:33<08:3 
86%|█████████████████████████████████████████████████████▍        | 28864/33480 [52:36<08:2 
86%|█████████████████████████████████████████████████████▌        | 28896/33480 [52:40<08:2 
86%|█████████████████████████████████████████████████████▌        | 28928/33480 [52:43<08:2 
86%|█████████████████████████████████████████████████████▋        | 28960/33480 [52:47<08:1 
87%|█████████████████████████████████████████████████████▋        | 28992/33480 [52:50<08:1 
87%|█████████████████████████████████████████████████████▋        | 29024/33480 [52:54<08:1 
87%|█████████████████████████████████████████████████████▊        | 29056/33480 [52:57<08:0 
87%|█████████████████████████████████████████████████████▊        | 29088/33480 [53:01<08:0 
87%|█████████████████████████████████████████████████████▉        | 29120/33480 [53:04<08:0 
87%|█████████████████████████████████████████████████████▉        | 29152/33480 [53:08<08:0 
87%|██████████████████████████████████████████████████████        | 29184/33480 [53:12<08:1 
87%|██████████████████████████████████████████████████████        | 29216/33480 [53:16<08:1 
87%|██████████████████████████████████████████████████████▏       | 29248/33480 [53:19<08:0 
87%|██████████████████████████████████████████████████████▏       | 29280/33480 [53:23<08:2 
88%|██████████████████████████████████████████████████████▎       | 29312/33480 [53:27<08:1 
88%|██████████████████████████████████████████████████████▎       | 29344/33480 [53:31<08:0 
88%|██████████████████████████████████████████████████████▍       | 29376/33480 [53:34<07:4 
88%|██████████████████████████████████████████████████████▍       | 29407/33480 [53:34<05:3 
88%|██████████████████████████████████████████████████████▍       | 29417/33480 [53:38<07:4 
88%|██████████████████████████████████████████████████████▌       | 29440/33480 [53:41<08:2 
88%|██████████████████████████████████████████████████████▌       | 29472/33480 [53:45<07:5 
88%|██████████████████████████████████████████████████████▋       | 29504/33480 [53:48<07:4 
88%|██████████████████████████████████████████████████████▋       | 29536/33480 [53:52<07:2 
88%|██████████████████████████████████████████████████████▊       | 29568/33480 [53:55<07:2 
88%|██████████████████████████████████████████████████████▊       | 29600/33480 [53:59<07:1 
89%|██████████████████████████████████████████████████████▊       | 29632/33480 [54:03<07:1 
89%|██████████████████████████████████████████████████████▉       | 29664/33480 [54:07<07:2 
89%|██████████████████████████████████████████████████████▉       | 29696/33480 [54:10<07:2 
89%|███████████████████████████████████████████████████████       | 29728/33480 [54:14<07:1 
89%|███████████████████████████████████████████████████████       | 29760/33480 [54:18<07:0 
89%|███████████████████████████████████████████████████████▏      | 29792/33480 [54:21<07:0 
89%|███████████████████████████████████████████████████████▏      | 29824/33480 [54:25<07:0 
89%|███████████████████████████████████████████████████████▎      | 29850/33480 [54:25<05:1 
89%|███████████████████████████████████████████████████████▎      | 29859/33480 [54:29<07:3 
89%|███████████████████████████████████████████████████████▎      | 29888/33480 [54:33<07:4 
89%|███████████████████████████████████████████████████████▍      | 29920/33480 [54:37<07:1 
89%|███████████████████████████████████████████████████████▍      | 29952/33480 [54:40<07:0 
90%|███████████████████████████████████████████████████████▌      | 29984/33480 [54:44<06:4 
90%|███████████████████████████████████████████████████████▌      | 30016/33480 [54:47<06:3 
90%|███████████████████████████████████████████████████████▋      | 30048/33480 [54:51<06:3 
90%|███████████████████████████████████████████████████████▋      | 30080/33480 [54:55<06:2 
90%|███████████████████████████████████████████████████████▊      | 30112/33480 [54:58<06:1 
90%|███████████████████████████████████████████████████████▊      | 30144/33480 [55:01<06:1 
90%|███████████████████████████████████████████████████████▉      | 30176/33480 [55:05<06:0 
90%|███████████████████████████████████████████████████████▉      | 30208/33480 [55:09<06:0 
90%|████████████████████████████████████████████████████████      | 30240/33480 [55:12<06:0 
90%|████████████████████████████████████████████████████████      | 30272/33480 [55:16<06:0 
91%|████████████████████████████████████████████████████████      | 30304/33480 [55:20<06:1 
91%|████████████████████████████████████████████████████████▏     | 30336/33480 [55:24<06:1 
91%|████████████████████████████████████████████████████████▏     | 30368/33480 [55:28<06:0 
91%|████████████████████████████████████████████████████████▎     | 30399/33480 [55:28<04:2 
91%|████████████████████████████████████████████████████████▎     | 30409/33480 [55:32<06:0 
91%|████████████████████████████████████████████████████████▎     | 30432/33480 [55:35<06:3 
91%|████████████████████████████████████████████████████████▍     | 30464/33480 [55:39<06:1 
91%|████████████████████████████████████████████████████████▍     | 30496/33480 [55:42<05:5 
91%|████████████████████████████████████████████████████████▌     | 30514/33480 [55:42<04:4 
91%|████████████████████████████████████████████████████████▌     | 30528/33480 [55:46<06:1 
91%|████████████████████████████████████████████████████████▌     | 30560/33480 [55:50<05:5 
91%|████████████████████████████████████████████████████████▋     | 30592/33480 [55:53<05:4 
91%|████████████████████████████████████████████████████████▋     | 30624/33480 [55:57<05:3 
92%|████████████████████████████████████████████████████████▊     | 30656/33480 [56:01<05:2 
92%|████████████████████████████████████████████████████████▊     | 30688/33480 [56:04<05:1 
92%|████████████████████████████████████████████████████████▉     | 30720/33480 [56:08<05:1 
92%|████████████████████████████████████████████████████████▉     | 30752/33480 [56:11<05:0 
92%|█████████████████████████████████████████████████████████     | 30783/33480 [56:11<03:3 
92%|█████████████████████████████████████████████████████████     | 30794/33480 [56:15<04:5 
92%|█████████████████████████████████████████████████████████     | 30816/33480 [56:18<05:2 
92%|█████████████████████████████████████████████████████████▏    | 30848/33480 [56:22<05:0 
92%|█████████████████████████████████████████████████████████▏    | 30880/33480 [56:25<04:5 
92%|█████████████████████████████████████████████████████████▏    | 30912/33480 [56:28<04:4 
92%|█████████████████████████████████████████████████████████▎    | 30944/33480 [56:32<04:4 
93%|█████████████████████████████████████████████████████████▎    | 30976/33480 [56:35<04:3 
93%|█████████████████████████████████████████████████████████▍    | 31008/33480 [56:39<04:3 
93%|█████████████████████████████████████████████████████████▍    | 31040/33480 [56:42<04:2 
93%|█████████████████████████████████████████████████████████▌    | 31072/33480 [56:46<04:2 
93%|█████████████████████████████████████████████████████████▌    | 31104/33480 [56:49<04:1 
93%|█████████████████████████████████████████████████████████▋    | 31136/33480 [56:53<04:1 
93%|█████████████████████████████████████████████████████████▋    | 31168/33480 [56:56<04:0 
93%|█████████████████████████████████████████████████████████▊    | 31200/33480 [57:00<04:0 
93%|█████████████████████████████████████████████████████████▊    | 31232/33480 [57:03<04:0 
93%|█████████████████████████████████████████████████████████▉    | 31264/33480 [57:07<03:5 
93%|█████████████████████████████████████████████████████████▉    | 31296/33480 [57:10<03:5 
94%|██████████████████████████████████████████████████████████    | 31328/33480 [57:13<03:5 
94%|██████████████████████████████████████████████████████████    | 31360/33480 [57:17<03:4 
94%|██████████████████████████████████████████████████████████▏   | 31392/33480 [57:20<03:4 
94%|██████████████████████████████████████████████████████████▏   | 31424/33480 [57:24<03:4 
94%|██████████████████████████████████████████████████████████▎   | 31456/33480 [57:27<03:3 
94%|██████████████████████████████████████████████████████████▎   | 31488/33480 [57:31<03:3 
94%|██████████████████████████████████████████████████████████▎   | 31520/33480 [57:34<03:2 
94%|██████████████████████████████████████████████████████████▍   | 31552/33480 [57:37<03:2 
94%|██████████████████████████████████████████████████████████▍   | 31584/33480 [57:41<03:2 
94%|██████████████████████████████████████████████████████████▌   | 31616/33480 [57:44<03:2 
95%|██████████████████████████████████████████████████████████▌   | 31648/33480 [57:48<03:1 
95%|██████████████████████████████████████████████████████████▋   | 31680/33480 [57:51<03:1 
95%|██████████████████████████████████████████████████████████▋   | 31712/33480 [57:55<03:1 
95%|██████████████████████████████████████████████████████████▊   | 31744/33480 [57:58<03:0 
95%|██████████████████████████████████████████████████████████▊   | 31776/33480 [58:02<03:0 
95%|██████████████████████████████████████████████████████████▉   | 31808/33480 [58:05<02:5 
95%|██████████████████████████████████████████████████████████▉   | 31840/33480 [58:09<02:5 
95%|███████████████████████████████████████████████████████████   | 31872/33480 [58:12<02:5 
95%|███████████████████████████████████████████████████████████   | 31904/33480 [58:15<02:4 
95%|███████████████████████████████████████████████████████████▏  | 31936/33480 [58:19<02:4 
95%|███████████████████████████████████████████████████████████▏  | 31968/33480 [58:22<02:4 
96%|███████████████████████████████████████████████████████████▎  | 32000/33480 [58:26<02:3 
96%|███████████████████████████████████████████████████████████▎  | 32032/33480 [58:29<02:3 
96%|███████████████████████████████████████████████████████████▍  | 32064/33480 [58:32<02:3 
96%|███████████████████████████████████████████████████████████▍  | 32096/33480 [58:36<02:2 
96%|███████████████████████████████████████████████████████████▍  | 32128/33480 [58:39<02:2 
96%|███████████████████████████████████████████████████████████▌  | 32160/33480 [58:43<02:2 
96%|███████████████████████████████████████████████████████████▌  | 32192/33480 [58:46<02:1 
96%|███████████████████████████████████████████████████████████▋  | 32224/33480 [58:50<02:1 
96%|███████████████████████████████████████████████████████████▋  | 32256/33480 [58:53<02:1 
96%|███████████████████████████████████████████████████████████▊  | 32288/33480 [58:56<02:0 
97%|███████████████████████████████████████████████████████████▊  | 32320/33480 [59:00<02:0 
97%|███████████████████████████████████████████████████████████▉  | 32352/33480 [59:03<02:0 
97%|███████████████████████████████████████████████████████████▉  | 32384/33480 [59:07<01:5 
97%|████████████████████████████████████████████████████████████  | 32416/33480 [59:10<01:5 
97%|████████████████████████████████████████████████████████████  | 32448/33480 [59:14<01:5 
97%|████████████████████████████████████████████████████████████▏ | 32480/33480 [59:17<01:4 
97%|████████████████████████████████████████████████████████████▏ | 32512/33480 [59:20<01:4 
97%|████████████████████████████████████████████████████████████▎ | 32544/33480 [59:24<01:4 
97%|████████████████████████████████████████████████████████████▎ | 32576/33480 [59:27<01:3 
97%|████████████████████████████████████████████████████████████▍ | 32608/33480 [59:31<01:3 
97%|████████████████████████████████████████████████████████████▍ | 32640/33480 [59:35<01:3 
98%|████████████████████████████████████████████████████████████▌ | 32672/33480 [59:38<01:3 
98%|████████████████████████████████████████████████████████████▌ | 32704/33480 [59:42<01:2 
98%|████████████████████████████████████████████████████████████▌ | 32736/33480 [59:46<01:2 
98%|████████████████████████████████████████████████████████████▋ | 32768/33480 [59:49<01:1 
98%|████████████████████████████████████████████████████████████▋ | 32800/33480 [59:53<01:1 
98%|████████████████████████████████████████████████████████████▊ | 32832/33480 [59:56<01:1 
98%|██████████████████████████████████████████████████████████▉ | 32864/33480 [1:00:00<01:0 
98%|██████████████████████████████████████████████████████████▉ | 32896/33480 [1:00:03<01:0 
98%|███████████████████████████████████████████████████████████ | 32928/33480 [1:00:07<01:0 
98%|███████████████████████████████████████████████████████████ | 32960/33480 [1:00:10<00:5 
99%|███████████████████████████████████████████████████████████▏| 32992/33480 [1:00:14<00:5 
99%|███████████████████████████████████████████████████████████▏| 33024/33480 [1:00:17<00:4 
99%|███████████████████████████████████████████████████████████▏| 33056/33480 [1:00:20<00:4 
99%|███████████████████████████████████████████████████████████▎| 33088/33480 [1:00:24<00:4 
99%|███████████████████████████████████████████████████████████▎| 33120/33480 [1:00:28<00:3 
99%|███████████████████████████████████████████████████████████▍| 33152/33480 [1:00:31<00:3 
99%|███████████████████████████████████████████████████████████▍| 33184/33480 [1:00:35<00:3 
99%|███████████████████████████████████████████████████████████▌| 33216/33480 [1:00:38<00:2 
99%|███████████████████████████████████████████████████████████▌| 33248/33480 [1:00:42<00:2 
99%|███████████████████████████████████████████████████████████▋| 33280/33480 [1:00:45<00:2 
99%|███████████████████████████████████████████████████████████▋| 33312/33480 [1:00:49<00:1
100%|███████████████████████████████████████████████████████████▋| 33324/33480 [1:00:49<00:1
100%|███████████████████████████████████████████████████████████▊| 33344/33480 [1:00:53<00:1
100%|███████████████████████████████████████████████████████████▊| 33376/33480 [1:00:57<00:1
100%|███████████████████████████████████████████████████████████▊| 33408/33480 [1:01:00<00:0
100%|███████████████████████████████████████████████████████████▉| 33440/33480 [1:01:03<00:0
100%|███████████████████████████████████████████████████████████▉| 33472/33480 [1:01:07<00:0
100%|████████████████████████████████████████████████████████████| 33480/33480 [1:01:08<00:0
100%|████████████████████████████████████████████████████████████| 33480/33480 [1:01:08<00:00,  9.13it/s]
length:  29789
100%|█████████████████████████████████████████████████| 29789/29789 [54:18<00:00,  9.14it/s]
length2:  33480
100%|████████████████████████████████████████████████| 33480/33480 [02:41<00:00, 207.33it/s]
t:  [('recall_top1_correct_composition', 0.025089605734767026), 
     ('recall_top5_correct_composition', 0.08136200716845877), 
     ('recall_top10_correct_composition', 0.13273596176821983), 
     ('recall_top50_correct_composition', 0.3347670250896057), 
     ('recall_top100_correct_composition', 0.4379330943847073)
    ]
(cirr) (base) menjiayu@Laputa-in-Air tirg % 
'''