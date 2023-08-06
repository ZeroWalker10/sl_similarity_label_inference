import torch
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import preprocessing
import pdb
import numpy as np

def collect_training(pipeline, train_loader, device, 
                           cache_size, pca, train_pca, 
                           max_dims=2048*3, normalize=True,
                           victim='gradient', mixup_alpha=None):
    collect_data = None
    pca_cache = None
    all_labels = None

    pipeline.reset()
    while not pipeline.is_end():
      entry = pipeline.next()
      if victim == 'smash':
        entry.role.eval()
      else:
        entry.role.train()

    if victim == 'gradient':
      pipeline.reset()
      while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.on_epoch_start()

    for i, (imgs, labels) in enumerate(train_loader):
      pipeline.reset()
      while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.on_batch_start()

      if mixup_alpha is not None and victim == 'gradient':
        # mixup training
        lamb = np.random.beta(mixup_alpha, mixup_alpha)
        indexes = torch.randperm(imgs.size(0))
        imgs = lamb * imgs + (1 - lamb) * imgs[indexes, :]
        ind_labels = labels[indexes]

        imgs = imgs.to(device)
        labels = labels.to(device)
        indexes = indexes.to(device)
      else:
        imgs = imgs.to(device)
        labels = labels.to(device)

      pipeline.reset()
      inputs = imgs
      # forward
      if victim == 'smash':
        with torch.no_grad():
          while not pipeline.is_end():
            entry = pipeline.next()
            outputs = entry.role.forward(inputs)
            inputs = outputs
            if entry.role.step != -1:
              # not last step
              batch_smashes = outputs.detach().cpu().numpy().reshape(len(imgs), -1).astype('float32')
              if batch_smashes.shape[1] > max_dims:
                if train_pca:
                  if pca_cache is None:
                    pca_cache = batch_smashes
                  else:
                    pca_cache = np.concatenate([pca_cache, batch_smashes]).astype('float32')

                  if len(pca_cache) >= cache_size:
                    pca.partial_fit(pca_cache.astype('float32'))
                    pca_cache = None
                else:
                  batch_smashes = pca.transform(batch_smashes.astype('float32'))
              else:
                train_pca = False

              if not train_pca:
                if normalize:
                  batch_smashes = preprocessing.normalize(batch_smashes).astype('float32')

                if collect_data is None:
                  collect_data = batch_smashes
                else:
                  collect_data = np.concatenate([collect_data, batch_smashes])
      else:
        while not pipeline.is_end():
          entry = pipeline.next()
          outputs = entry.role.forward(inputs)
          inputs = outputs

      predictions = outputs.to(device)

      # backward
      pipeline.r_reset()
      inputs = labels
      while victim != 'smash' and not pipeline.r_is_end():
        entry = pipeline.r_next()

        if entry.role.step == -1:
          if mixup_alpha is not None:
            outputs = entry.role.backward(inputs, indexes, lamb)
          else:
            outputs = entry.role.backward(inputs)
        else:
          outputs = entry.role.backward(inputs)

        if entry.role.step == -1:
          batch_grads = outputs.detach().cpu().numpy().reshape(len(imgs), -1).astype('float32')
          if batch_grads.shape[1] > max_dims:
            if train_pca:
              if pca_cache is None:
                pca_cache = batch_grads
              else:
                pca_cache = np.concatenate([pca_cache, batch_grads]).astype('float32')

              if len(pca_cache) >= cache_size:
                pca.partial_fit(pca_cache.astype('float32'))
                pca_cache = None
            else:
              batch_grads = pca.transform(batch_grads.astype('float32'))
          else:
            train_pca = False

          if not train_pca:
            if normalize:
              batch_grads = preprocessing.normalize(batch_grads)

            if collect_data is None:
              collect_data = batch_grads
            else:
              collect_data = np.concatenate([collect_data, batch_grads])
        inputs = outputs

      if all_labels is None:
        all_labels = labels.cpu().numpy()
      else:
        all_labels = np.concatenate([all_labels, labels.cpu().numpy()])

      if device.type == 'cuda':
        torch.cuda.empty_cache()

      if victim == 'gradient':
        pipeline.reset()
        while not pipeline.is_end():
          entry = pipeline.next()
          entry.role.on_batch_end()

    if victim == 'gradient':
      pipeline.reset()
      while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.on_epoch_end()

    return collect_data, all_labels

def collect_training_fake(pipeline, train_loader, device, 
                           cache_size, pca, train_pca, 
                           max_dims=2048*3, normalize=True,
                           victim='gradient', mixup_alpha=None):
    collect_data = None
    pca_cache = None
    all_labels = None

    pipeline.reset()
    while not pipeline.is_end():
      entry = pipeline.next()
      if victim == 'smash':
        entry.role.eval()
      else:
        entry.role.train()

    if victim == 'gradient':
      pipeline.reset()
      while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.on_epoch_start()

    for i, (imgs, labels, fake_labels) in enumerate(train_loader):
      pipeline.reset()
      while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.on_batch_start()

      if mixup_alpha is not None and victim == 'gradient':
        # mixup training
        lamb = np.random.beta(mixup_alpha, mixup_alpha)
        indexes = torch.randperm(imgs.size(0))
        imgs = lamb * imgs + (1 - lamb) * imgs[indexes, :]
        ind_labels = labels[indexes]

        imgs = imgs.to(device)
        labels = labels.to(device)
        indexes = indexes.to(device)
      else:
        imgs = imgs.to(device)
        labels = labels.to(device)

      fake_labels = fake_labels.to(device)

      pipeline.reset()
      inputs = imgs
      # forward
      if victim == 'smash':
        with torch.no_grad():
          while not pipeline.is_end():
            entry = pipeline.next()
            outputs = entry.role.forward(inputs)
            inputs = outputs
            if entry.role.step != -1:
              # not last step
              batch_smashes = outputs.detach().cpu().numpy().reshape(len(imgs), -1).astype('float32')
              if batch_smashes.shape[1] > max_dims:
                if train_pca:
                  if pca_cache is None:
                    pca_cache = batch_smashes
                  else:
                    pca_cache = np.concatenate([pca_cache, batch_smashes]).astype('float32')

                  if len(pca_cache) >= cache_size:
                    pca.partial_fit(pca_cache.astype('float32'))
                    pca_cache = None
                else:
                  batch_smashes = pca.transform(batch_smashes.astype('float32'))
              else:
                train_pca = False

              if not train_pca:
                if normalize:
                  batch_smashes = preprocessing.normalize(batch_smashes).astype('float32')

                if collect_data is None:
                  collect_data = batch_smashes
                else:
                  collect_data = np.concatenate([collect_data, batch_smashes])
      else:
        while not pipeline.is_end():
          entry = pipeline.next()
          outputs = entry.role.forward(inputs)
          inputs = outputs

      predictions = outputs.to(device)

      # backward
      pipeline.r_reset()
      inputs = labels
      fake_inputs = fake_labels
      while victim != 'smash' and not pipeline.r_is_end():
        entry = pipeline.r_next()

        if entry.role.step == -1:
          if mixup_alpha is not None:
            outputs = entry.role.backward(inputs, fake_inputs, indexes, lamb)
          else:
            outputs = entry.role.backward(inputs, fake_inputs)
        else:
          outputs = entry.role.backward(inputs)

        if entry.role.step == -1:
          batch_grads = outputs.detach().cpu().numpy().reshape(len(imgs), -1).astype('float32')
          if batch_grads.shape[1] > max_dims:
            if train_pca:
              if pca_cache is None:
                pca_cache = batch_grads
              else:
                pca_cache = np.concatenate([pca_cache, batch_grads]).astype('float32')

              if len(pca_cache) >= cache_size:
                pca.partial_fit(pca_cache.astype('float32'))
                pca_cache = None
            else:
              batch_grads = pca.transform(batch_grads.astype('float32'))
          else:
            train_pca = False

          if not train_pca:
            if normalize:
              batch_grads = preprocessing.normalize(batch_grads)

            if collect_data is None:
              collect_data = batch_grads
            else:
              collect_data = np.concatenate([collect_data, batch_grads])
        inputs = outputs

      if all_labels is None:
        all_labels = labels.cpu().numpy()
      else:
        all_labels = np.concatenate([all_labels, labels.cpu().numpy()])

      if device.type == 'cuda':
        torch.cuda.empty_cache()

      if victim == 'gradient':
        pipeline.reset()
        while not pipeline.is_end():
          entry = pipeline.next()
          entry.role.on_batch_end()

    if victim == 'gradient':
      pipeline.reset()
      while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.on_epoch_end()

    return collect_data, all_labels
