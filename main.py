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
from tensorboardX import SummaryWriter
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
  model, optimizer = create_model_and_optimizer(opt, trainset.get_all_texts())
  
  # Ensure input is on the same device as the model
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  '''1st try - start'''

  # # ### start the training
  # # train_loop(opt, logger, trainset, testset, model, optimizer)
  # # logger.close()

  # '''load the pretrained model'''
  # checkpoint = torch.load('checkpoint_fashion200k.pth',map_location=torch.device('cpu'),weights_only=False)
  # model.load_state_dict(checkpoint['model_state_dict'])

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
  python main.py --dataset=fashion200k --dataset_path=./Fashion200k \
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
  t = test_retrieval.test(opt, model, trainset)
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
(base) menjiayu@MacBookAir tirg % python main.py --dataset=fashion200k --dataset_path=./Fashion200k \
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