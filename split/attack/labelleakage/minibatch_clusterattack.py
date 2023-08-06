import torch
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans
from sklearn import preprocessing
from Bio.Cluster import kcluster
import pdb
import numpy as np

def clusterattack_training(pipeline, train_loader, labels_set, device, pca_components=-1, distance='euclidean', victim='gradient',
                           known_labels=None, known_smashes=None, known_labels_init=False, known_labels_offset=-1,
                           mixup_alpha=None):
    '''
    distance support: euclidean, cosine_smiliary, and cosine_distance
    victim support: gradient and smash
    '''
    all_grads = []
    all_smashes = []
    all_labels = []
    pre_known = False

    known_gradients = []
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
        # pdb.set_trace()
        if batch_size is None:
          batch_size = len(imgs)

        if mixup_alpha is not None:
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
                batch_smashes = outputs.detach().cpu().numpy().reshape(len(imgs), -1)
                if batch_smashes.shape[1] > max_features:
                  if fp is None:
                    fp = open(tmp_file, 'wb')
                  np.save(fp, batch_smashes)
                  num_writes += 1
                  # all_smashes.append(batch_smashes[:, :max_features].tolist())
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
              if mixup_alpha is not None:
                outputs = entry.role.backward(inputs, indexes, lamb)
              else:
                outputs = entry.role.backward(inputs)
            else:
              outputs = entry.role.backward(inputs)

            if entry.role.step == -1:
                # last step
                # pdb.set_trace()
                if victim == 'gradient':
                    batch_grads = outputs.cpu().numpy().reshape(len(imgs), -1)
                    if batch_grads.shape[1] > max_features:
                      if fp is None:
                        fp = open(tmp_file, 'wb')
                      np.save(fp, batch_grads)
                      num_writes += 1
                      # all_grads.append(batch_grads[:, :max_features].tolist())
                    else:
                      all_grads.append(batch_grads)
            inputs = outputs

        # fill known labels randomly
        if i > known_labels_offset:
            for label in labels_set:
              if label.item() in known_labels:
                continue
              label_indexes = torch.nonzero(labels == label,as_tuple=True)[0]
              if len(label_indexes) == 0:
                continue
              # choose one grad randomly
              index = label_indexes[torch.randperm(len(label_indexes))][0]
              known_labels[label.item()] = offset + index.item()
              if victim == 'smash':
                known_smashes[label.item()] = batch_smashes[index.item()]
              else:
                known_gradients.append(batch_grads[index.item()])
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

    all_clusters = []
    if victim == 'gradient' and len(all_grads) == 0:
      # if known_labels_init is False:
      kmeans = MiniBatchKMeans(n_clusters=len(labels_set), init='k-means++')
      # elif known_labels_init is True:
      #  kmeans = MiniBatchKMeans(n_clusters=len(labels_set), init=np.vstack(known_gradients), n_init=1)

      for _ in range(20):
        with open(tmp_file, 'rb') as fp:
          for _ in range(num_writes):
            batch_grads = np.load(fp)
            kmeans.partial_fit(preprocessing.normalize(batch_grads))

      with open(tmp_file, 'rb') as fp:
        for _ in range(num_writes):
          batch_grads = np.load(fp)
          batch_pred_clusters = kmeans.predict(preprocessing.normalize(batch_grads))
          all_clusters.append(batch_pred_clusters)

    all_labels = np.concatenate(all_labels)
    all_clusters = np.concatenate(all_clusters)

    pred_labels = np.ones(len(all_clusters), dtype='int') * -1
    for label, index in known_labels.items():
      pred_labels[all_clusters == all_clusters[index]] = label

    if pre_known:
      pred_labels = pred_labels[:-len(known_smashes)]

    return pred_labels, all_labels, known_labels, known_smashes
