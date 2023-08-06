import torch
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import preprocessing
from Bio.Cluster import kcluster
import pdb
import numpy as np

def clusterattack_training(pipeline, train_loader, labels_set, device, pca_components=-1, distance='euclidean', victim='gradient',
                           known_labels=None, known_smashes=None, known_labels_init=False, known_labels_offset=-1):
    '''
    distance support: euclidean, cosine_smiliary, and cosine_distance
    victim support: gradient and smash
    '''
    all_grads = []
    all_smashes = []
    all_labels = []
    pre_known = False

    if known_labels is None:
      known_labels = {}

    if known_smashes is None:
      known_smashes = {}
    else:
      pre_known = True
    offset = 0
    tmp_file = './tmp.npy'
    fp = None
    num_writes = 0
    batch_size = None
    max_features = 10240
    for i, (imgs, labels, ground_truths) in enumerate(train_loader):
        if batch_size is None:
          batch_size = len(imgs)

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
                batch_smashes = outputs.detach().cpu().numpy().reshape(len(imgs), -1)
                if batch_smashes.shape[1] > max_features:
                  '''
                  if fp is None:
                    fp = open(tmp_file, 'wb')
                  np.save(fp, batch_smashes)
                  num_writes += 1
                  '''
                  all_smashes.append(batch_smashes[:, :max_features].tolist())
                else:
                  all_smashes.append(batch_smashes)
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
            outputs = entry.role.backward(inputs)
            if entry.role.step == -1:
                # last step
                # pdb.set_trace()
                if victim == 'gradient':
                    batch_grads = outputs.cpu().numpy().reshape(len(imgs), -1)
                    if batch_grads.shape[1] > max_features:
                      '''
                      if fp is None:
                        fp = open(tmp_file, 'wb')
                      np.save(fp, batch_grads)
                      num_writes += 1
                      '''
                      all_grads.append(batch_grads[:, :max_features].tolist())
                    else:
                      all_grads.append(batch_grads)
            inputs = outputs

        # fill known labels randomly
        if i > known_labels_offset:
            for label in labels_set:
              if label.item() in known_labels:
                continue
              label_indexes = torch.nonzero(ground_truths.cpu() == label.cpu(),as_tuple=True)[0]
              if len(label_indexes) == 0:
                continue
              # choose one grad randomly
              index = label_indexes[torch.randperm(len(label_indexes))][0]
              known_labels[label.item()] = offset + index.item()
              if victim == 'smash':
                known_smashes[label.item()] = batch_smashes[index.item()]
        all_labels.append(labels.cpu().numpy())
        offset += len(labels)

        # clear GPU memory
        del imgs, labels
        if victim == 'gradient':
          del predictions

        if device.type == 'cuda':
          torch.cuda.empty_cache()

    if fp is not None:
      fp.close()

    if victim == 'gradient' and len(all_grads) > 0 and pca_components > 0 and pca_components < len(all_grads[0]):
      all_grads = np.concatenate(all_grads)
      all_ngrads = all_grads

      pca = PCA(n_components=pca_components)
      all_ngrads = pca.fit_transform(all_ngrads)
    elif victim == 'gradient' and len(all_grads) == 0:
      pca = IncrementalPCA(n_components=None, batch_size=batch_size)
      with open(tmp_file, 'rb') as fp:
        for _ in range(num_writes):
          batch_grads = np.load(fp)
          pca.partial_fit(batch_grads)
      with open(tmp_file, 'rb') as fp:
        for _ in range(num_writes):
          batch_grads = np.load(fp)
          new_batch_grads = pca.transform(batch_grads)
          all_grads.append(new_batch_grads)
      all_ngrads = np.concatenate(all_grads)
    elif victim == 'gradient':
      all_ngrads = np.concatenate(all_grads)

    if victim == 'smash' and len(all_smashes) > 0 and pca_components > 0 and pca_components < len(all_smashes[0]):
      all_smashes = np.concatenate(all_smashes)
      if pre_known:
        for k, item in known_smashes.items():
          known_labels[k] = len(all_smashes)
          all_smashes = np.concatenate([all_smashes, item[np.newaxis, :]])
      all_nsmashes = all_smashes

      pca = PCA(n_components=pca_components)
      all_nmashes = pca.fit_transform(all_nsmashes)
    elif victim == 'smash' and len(all_smashes) == 0:
      pca = IncrementalPCA(n_components=None, batch_size=batch_size)
      index = 0
      with open(tmp_file, 'rb') as fp:
        for i in range(num_writes):
          batch_smashes = np.load(fp)
          index += len(batch_smashes)

          if i + 1 == num_writes and pre_known:
            for k, item in known_smashes:
              known_smashes[k] = index
              batch_smashes = np.concatenate([batch_smashes, item[np.newaxis, :]])
          pca.partial_fit(batch_smashes)
      with open(tmp_file, 'rb') as fp:
        for i in range(num_writes):
          batch_smashes = np.load(fp)
          if i + 1 == num_writes and pre_known:
            for k, item in known_smashes:
              batch_smashes = np.concatenate([batch_smashes, item[np.newaxis, :]])
          new_batch_smashes = pca.transform(batch_smashes)
          all_smashes.append(new_batch_smashes)
      all_nsmashes = np.concatenate(all_smashes)
    elif victim == 'smash':
      all_smashes = np.concatenate(all_smashes)
      if pre_known:
        for k, item in known_smashes.items():
          known_labels[k] = len(all_smashes)
          all_smashes = np.concatenate([all_smashes, item[np.newaxis, :]])
      all_nsmashes = all_smashes

    all_labels = np.concatenate(all_labels)

    if distance == 'euclidean':
        if victim == 'gradient' and known_labels_init is False:
          kmeans = KMeans(n_clusters=len(labels_set), init='k-means++').fit(all_ngrads)
        elif victim == 'gradient' and known_labels_init is True:
          init_grads = all_ngrads[list(known_labels.values())]
          kmeans = KMeans(n_clusters=len(labels_set), init=init_grads, n_init=1).fit(all_ngrads)
        elif victim == 'smash' and known_labels_init is False:
          kmeans = KMeans(n_clusters=len(labels_set), init='k-means++').fit(all_nsmashes)
        elif victim == 'smash' and known_labels_init is True:
          init_smashes = all_nsmashes[list(known_labels.values())]
          kmeans = KMeans(n_clusters=len(labels_set), init=init_smashes, n_init=1).fit(all_nsmashes)
        clusters = kmeans.labels_
    elif distance == 'cosine_distance':
        if victim == 'gradient':
          all_ngrads = preprocessing.normalize(all_ngrads)
        elif victim == 'smash':
          all_nsmashes = preprocessing.normalize(all_nsmashes)

        if victim == 'gradient' and known_labels_init is False:
          kmeans = KMeans(n_clusters=len(labels_set), init='k-means++').fit(all_ngrads)
        elif victim == 'gradient' and known_labels_init is True:
          init_grads = all_ngrads[list(known_labels.values())]
          kmeans = KMeans(n_clusters=len(labels_set), init=init_grads, n_init=1).fit(all_ngrads)
        elif victim == 'smash' and known_labels_init is False:
          kmeans = KMeans(n_clusters=len(labels_set), init='k-means++').fit(all_nsmashes)
        elif victim == 'smash' and known_labels_init is True:
          init_smashes = all_nsmashes[list(known_labels.values())]
          kmeans = KMeans(n_clusters=len(labels_set), init=init_smashes, n_init=1).fit(all_nsmashes)
        clusters = kmeans.labels_

    pred_labels = np.ones(len(clusters), dtype='int') * -1
    for label, index in known_labels.items():
      pred_labels[clusters == clusters[index]] = label

    if pre_known:
      pred_labels = pred_labels[:-len(known_smashes)]

    return pred_labels, all_labels, known_labels, known_smashes

def clusterattack_valid(pipeline, train_loader, labels_set, device, pca_components=-1, distance='euclidean', victim='gradient',
                           known_labels=None, known_smashes=None, known_labels_init=False, known_labels_offset=-1):
    '''
    distance support: euclidean, cosine_smiliary, and cosine_distance
    victim support: gradient and smash
    '''
    all_grads = []
    all_smashes = []
    all_labels = []
    pre_known = False

    if known_labels is None:
      known_labels = {}

    if known_smashes is None:
      known_smashes = {}
    else:
      pre_known = True
    offset = 0
    tmp_file = './tmp.npy'
    fp = None
    num_writes = 0
    batch_size = None
    max_features = 10240
    for i, (imgs, labels) in enumerate(train_loader):
        if batch_size is None:
          batch_size = len(imgs)

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
                batch_smashes = outputs.detach().cpu().numpy().reshape(len(imgs), -1)
                if batch_smashes.shape[1] > max_features:
                  '''
                  if fp is None:
                    fp = open(tmp_file, 'wb')
                  np.save(fp, batch_smashes)
                  num_writes += 1
                  '''
                  all_smashes.append(batch_smashes[:, :max_features].tolist())
                else:
                  all_smashes.append(batch_smashes)
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
            outputs = entry.role.backward(inputs)
            if entry.role.step == -1:
                # last step
                # pdb.set_trace()
                if victim == 'gradient':
                    batch_grads = outputs.cpu().numpy().reshape(len(imgs), -1)
                    if batch_grads.shape[1] > max_features:
                      '''
                      if fp is None:
                        fp = open(tmp_file, 'wb')
                      np.save(fp, batch_grads)
                      num_writes += 1
                      '''
                      all_grads.append(batch_grads[:, :max_features].tolist())
                    else:
                      all_grads.append(batch_grads)
            inputs = outputs

        # fill known labels randomly
        if i > known_labels_offset:
            for label in labels_set:
              if label.item() in known_labels:
                continue
              label_indexes = torch.nonzero(ground_truths.cpu() == label.cpu(),as_tuple=True)[0]
              if len(label_indexes) == 0:
                continue
              # choose one grad randomly
              index = label_indexes[torch.randperm(len(label_indexes))][0]
              known_labels[label.item()] = offset + index.item()
              if victim == 'smash':
                known_smashes[label.item()] = batch_smashes[index.item()]
        all_labels.append(labels.cpu().numpy())
        offset += len(labels)

        # clear GPU memory
        del imgs, labels
        if victim == 'gradient':
          del predictions

        if device.type == 'cuda':
          torch.cuda.empty_cache()

    if fp is not None:
      fp.close()

    if victim == 'gradient' and len(all_grads) > 0 and pca_components > 0 and pca_components < len(all_grads[0]):
      all_grads = np.concatenate(all_grads)
      all_ngrads = all_grads

      pca = PCA(n_components=pca_components)
      all_ngrads = pca.fit_transform(all_ngrads)
    elif victim == 'gradient' and len(all_grads) == 0:
      pca = IncrementalPCA(n_components=None, batch_size=batch_size)
      with open(tmp_file, 'rb') as fp:
        for _ in range(num_writes):
          batch_grads = np.load(fp)
          pca.partial_fit(batch_grads)
      with open(tmp_file, 'rb') as fp:
        for _ in range(num_writes):
          batch_grads = np.load(fp)
          new_batch_grads = pca.transform(batch_grads)
          all_grads.append(new_batch_grads)
      all_ngrads = np.concatenate(all_grads)
    elif victim == 'gradient':
      all_ngrads = np.concatenate(all_grads)

    if victim == 'smash' and len(all_smashes) > 0 and pca_components > 0 and pca_components < len(all_smashes[0]):
      all_smashes = np.concatenate(all_smashes)
      if pre_known:
        for k, item in known_smashes.items():
          known_labels[k] = len(all_smashes)
          all_smashes = np.concatenate([all_smashes, item[np.newaxis, :]])
      all_nsmashes = all_smashes

      pca = PCA(n_components=pca_components)
      all_nmashes = pca.fit_transform(all_nsmashes)
    elif victim == 'smash' and len(all_smashes) == 0:
      pca = IncrementalPCA(n_components=None, batch_size=batch_size)
      index = 0
      with open(tmp_file, 'rb') as fp:
        for i in range(num_writes):
          batch_smashes = np.load(fp)
          index += len(batch_smashes)

          if i + 1 == num_writes and pre_known:
            for k, item in known_smashes:
              known_smashes[k] = index
              batch_smashes = np.concatenate([batch_smashes, item[np.newaxis, :]])
          pca.partial_fit(batch_smashes)
      with open(tmp_file, 'rb') as fp:
        for i in range(num_writes):
          batch_smashes = np.load(fp)
          if i + 1 == num_writes and pre_known:
            for k, item in known_smashes:
              batch_smashes = np.concatenate([batch_smashes, item[np.newaxis, :]])
          new_batch_smashes = pca.transform(batch_smashes)
          all_smashes.append(new_batch_smashes)
      all_nsmashes = np.concatenate(all_smashes)
    elif victim == 'smash':
      all_smashes = np.concatenate(all_smashes)
      if pre_known:
        for k, item in known_smashes.items():
          known_labels[k] = len(all_smashes)
          all_smashes = np.concatenate([all_smashes, item[np.newaxis, :]])
      all_nsmashes = all_smashes

    all_labels = np.concatenate(all_labels)

    if distance == 'euclidean':
        if victim == 'gradient' and known_labels_init is False:
          kmeans = KMeans(n_clusters=len(labels_set), init='k-means++').fit(all_ngrads)
        elif victim == 'gradient' and known_labels_init is True:
          init_grads = all_ngrads[list(known_labels.values())]
          kmeans = KMeans(n_clusters=len(labels_set), init=init_grads, n_init=1).fit(all_ngrads)
        elif victim == 'smash' and known_labels_init is False:
          kmeans = KMeans(n_clusters=len(labels_set), init='k-means++').fit(all_nsmashes)
        elif victim == 'smash' and known_labels_init is True:
          init_smashes = all_nsmashes[list(known_labels.values())]
          kmeans = KMeans(n_clusters=len(labels_set), init=init_smashes, n_init=1).fit(all_nsmashes)
        clusters = kmeans.labels_
    elif distance == 'cosine_distance':
        if victim == 'gradient':
          all_ngrads = preprocessing.normalize(all_ngrads)
        elif victim == 'smash':
          all_nsmashes = preprocessing.normalize(all_nsmashes)

        if victim == 'gradient' and known_labels_init is False:
          kmeans = KMeans(n_clusters=len(labels_set), init='k-means++').fit(all_ngrads)
        elif victim == 'gradient' and known_labels_init is True:
          init_grads = all_ngrads[list(known_labels.values())]
          kmeans = KMeans(n_clusters=len(labels_set), init=init_grads, n_init=1).fit(all_ngrads)
        elif victim == 'smash' and known_labels_init is False:
          kmeans = KMeans(n_clusters=len(labels_set), init='k-means++').fit(all_nsmashes)
        elif victim == 'smash' and known_labels_init is True:
          init_smashes = all_nsmashes[list(known_labels.values())]
          kmeans = KMeans(n_clusters=len(labels_set), init=init_smashes, n_init=1).fit(all_nsmashes)
        clusters = kmeans.labels_

    pred_labels = np.ones(len(clusters), dtype='int') * -1
    for label, index in known_labels.items():
      pred_labels[clusters == clusters[index]] = label

    if pre_known:
      pred_labels = pred_labels[:-len(known_smashes)]

    return pred_labels, all_labels, known_labels, known_smashes
