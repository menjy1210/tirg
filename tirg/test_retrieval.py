# Copyright 2018 Google Inc. All Rights Reserved.
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

"""Evaluates the retrieval model."""
import numpy as np
import torch
from tqdm import tqdm as tqdm


def test(opt, model, testset):
  """Tests a model over the given testset."""
  model.eval() # close up the dropout layers
  test_queries = testset.get_test_queries()

  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  if test_queries: # if there are test queries
    # compute test query features
    imgs = []
    mods = []
    print('len(test_queries): ',len(test_queries))
    for t in tqdm(test_queries):
      imgs += [testset.get_img(t['source_img_id'])]
      mods += [t['mod']['str']]
      if len(imgs) >= opt.batch_size or t is test_queries[-1]:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        # imgs = torch.autograd.Variable(imgs).cuda()
        imgs = torch.autograd.Variable(imgs).cpu()
        '''
        extract the composed feature 
        of the corresponding img and text in the test_query
        '''
        f = model.compose_img_text(imgs, mods).data.cpu().numpy()
        all_queries += [f]
        imgs = []
        mods = []
    all_queries = np.concatenate(all_queries)
    all_target_captions = [t['target_caption'] for t in test_queries]

    # compute all image features
    imgs = []
    print('length: ', len(testset.imgs))
    for i in tqdm(range(len(testset.imgs))):
      imgs += [testset.get_img(i)]
      if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs).cpu()# imgs = torch.autograd.Variable(imgs).cuda()
        '''
        extract the feature of all the imgs in the dataset
        '''
        imgs = model.extract_img_feature(imgs).data.cpu().numpy()
        all_imgs += [imgs]
        imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [img['captions'][0] for img in testset.imgs]

  else:# if there are no test queries
    # use training queries to approximate training retrieval performance
    imgs0 = []
    imgs = []
    mods = []
    for i in range(10000):
      item = testset[i]
      imgs += [item['source_img_data']]
      mods += [item['mod']['str']]
      if len(imgs) > opt.batch_size or i == 9999:
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)
        # f = model.compose_img_text(imgs.cuda(), mods).data.cpu().numpy()
        f = model.compose_img_text(imgs.cpu(), mods).data.cpu().numpy()
        print('---\nf.shape: ', f.shape)
        all_queries += [f]
        imgs = []
        mods = []
      imgs0 += [item['target_img_data']]
      if len(imgs0) > opt.batch_size or i == 9999:
        imgs0 = torch.stack(imgs0).float()
        imgs0 = torch.autograd.Variable(imgs0)
        # imgs0 = model.extract_img_feature(imgs0.cuda()).data.cpu().numpy()
        imgs0 = model.extract_img_feature(imgs0.cpu()).data.cpu().numpy()
        all_imgs += [imgs0]
        imgs0 = []
      all_captions += [item['target_caption']]
      all_target_captions += [item['target_caption']]
    all_imgs = np.concatenate(all_imgs)
    all_queries = np.concatenate(all_queries)

  # feature normalization
  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  nn_result = []
  print('length2: ', all_queries.shape[0])
  for i in tqdm(range(all_queries.shape[0])):
    sims = all_queries[i:(i+1), :].dot(all_imgs.T)
    if test_queries:
      sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
    nn_result.append(np.argsort(-sims[0, :])[:110])

  # compute recalls
  out = []
  nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]

  print('---\n nn_result.shape:', len(nn_result),',', len(nn_result[0]))
  '''
  nn_result.shape: 33480 , 110
  '''

  print('---\n all_target_captions.shape:', len(all_target_captions),',', len(all_target_captions[0]))
  '''
  all_target_captions.shape: 33480 , 6
  '''
  # save the results in a txt file
  with open('nn_results.txt', "w") as file:
    for row in nn_result:
        # 将每个元素转为字符串并用空格连接，最后加换行符
        line = " ".join(map(str, row)) + "\n"
        file.write(line)
  file.close()

  with open('all_target_captions.txt', "w") as file:
    for row in all_target_captions:
        # 将每个元素转为字符串并用空格连接，最后加换行符
        line = " ".join(map(str, row)) + "\n"
        file.write(line)
  file.close()

  for k in [1, 5, 10, 50, 100]: # recall - top k results
    r = 0.0
    for i, nns in enumerate(nn_result):
      # i: the i-th test query
      # nns: the sorted neighbors of the i-th query
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(nn_result)
    out += [('recall_top' + str(k) + '_correct_composition', r)]

    if opt.dataset == 'mitstates':
      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_adj', r)]

      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_noun', r)]

  return out
