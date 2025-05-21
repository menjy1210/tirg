'''
based on the example of Zero-Shot Prediction on Github
'''

import os
import clip
import torch
import torchvision  
import tqdm
import numpy as np

import datasets
# from torchvision.datasets import CIFAR100
# use the example from dataset fashion200K

def tirg_load_dataset(path):
    trainset = datasets.Fashion200k(
        path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.Fashion200k(
        path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    return trainset, testset

# def tirg_test(batc_size, model, testset):
#   return result

if __name__ == '__main__':
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Load the dataset ‘fashion200K’
    trainset, testset = tirg_load_dataset(path='./Fashion200k')

    
    '''
    prepare the inputs
    ---
    image, class_id = cifar100[3637]
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
    '''
    batch_size = 32
    test_queries = testset.get_test_queries()

    # calculate all the test query features
    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []

    # compute test query features
    imgs = []
    mods = []
    print('len(test_queries): ',len(test_queries))
    for t in tqdm(test_queries):
      imgs += [testset.get_img(t['source_img_id'])]
      mods += [t['mod']['str']]
      if len(imgs) >= batch_size or t is test_queries[-1]:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        # imgs = torch.autograd.Variable(imgs).cuda()
        imgs = torch.autograd.Variable(imgs).cpu()
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
      if len(imgs) >= batch_size or i == len(testset.imgs) - 1:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        # imgs = torch.autograd.Variable(imgs).cuda()
        imgs = torch.autograd.Variable(imgs).cpu()
        imgs = model.extract_img_feature(imgs).data.cpu().numpy()
        all_imgs += [imgs]
        imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [img['captions'][0] for img in testset.imgs]

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

    '''
    Pick the top 5 most similar labels for the image
    ---
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)
    '''
    # match test queries to target images, get nearest neighbors
    nn_result = []
    print('length2: ', all_queries.shape[0])
    for i in tqdm(range(all_queries.shape[0])):
        sims = all_queries[i:(i+1), :].dot(all_imgs.T)
        if test_queries:
            sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
        nn_result.append(np.argsort(-sims[0, :])[:110])


    '''
    Print the result
    ---
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
    
    '''
    # compute recalls
    out = []
    nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
    for k in [1, 5, 10, 50, 100]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if all_target_captions[i] in nns[:k]:
                r += 1
        r /= len(nn_result)
        out += [('recall_top' + str(k) + '_correct_composition', r)]
