import torch
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import preprocessing
import pdb
import numpy as np

def select_features_entropy(orig_features, max_dims):
  def _entropy(xs):
    return -np.sum(xs[xs > 0.0] * np.log2(xs[xs > 0.0]))

  orig_features_abs = np.abs(orig_features)
  features_entropy = np.apply_along_axis(_entropy, 0, orig_features_abs)
  selected_dims = np.sort(np.argsort(features_entropy)[-max_dims:])
  return selected_dims

def select_features_vt(orig_features, max_dims):
  vt = VarianceThreshold()
  vt.fit(X=orig_features)
  selected_dims = np.sort(np.argsort(vt.variances_)[-max_dims:])
  return selected_dims

def clusterattack_training(pipeline, train_loader, labels_set, device, pca_components=-1, distance='euclidean', victim='gradient',
                           known_labels=None, known_smashes=None, known_labels_init=False, known_labels_offset=-1,
                           mixup_alpha=None):
    '''
    distance support: euclidean, and cosine_distance
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
    
    batch_size = None
    if pca_components <= 0:
      max_features = 10240
    else:
      max_features = pca_components

    n_class = len(labels_set)
    if n_class <= 10:
      max_samples = n_class * 10
    elif n_class <= 100:
      max_samples = n_class * 5
    else:
      max_samples = len(labels_set) * 2
    selected_candidates = None
    selected_dims = None

    # training mode or eval mode
    pipeline.reset()
    while not pipeline.is_end():
      entry = pipeline.next()
      if victim == 'smash':
        entry.role.eval()
      else:
        entry.role.train()

    for i, (imgs, labels) in enumerate(train_loader):
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
                  if selected_dims is None:
                    if selected_candidates is None:
                      selected_candidates = batch_smashes
                    else:
                      selected_candidates = np.concatenate([selected_candidates,
                                                            batch_smashes])
                  else:
                    all_smashes.append(batch_smashes[:, selected_dims].tolist())
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
                    batch_grads = outputs.detach().cpu().numpy().reshape(len(imgs), -1)
                    if batch_grads.shape[1] > max_features:
                      if selected_dims is None:
                        if selected_candidates is None:
                          selected_candidates = batch_grads
                        else:
                          selected_candidates = np.concatenate([selected_candidates,
                                                              batch_grads])
                      else:
                        all_grads.append(batch_grads[:, selected_dims].tolist())
                    else:
                      all_grads.append(batch_grads)
            inputs = outputs

        if selected_candidates is not None and \
            len(selected_candidates) >= max_samples and \
          selected_dims is None:
          selected_dims = select_features_entropy(selected_candidates,
                                          max_features)
          if victim == 'smash':
            all_smashes = [selected_candidates[:, selected_dims].tolist()]
          else:
            all_grads = [selected_candidates[:, selected_dims].tolist()]

          selected_candidates = None

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
        all_labels.append(labels.cpu().numpy())
        offset += len(labels)

        # clear GPU memory
        del imgs, labels
        if victim == 'gradient':
          del predictions

        if device.type == 'cuda':
          torch.cuda.empty_cache()

    # fix known smashes
    if selected_dims is not None:
      for k in known_smashes:
        if len(known_smashes[k]) > max_features:
          known_smashes[k] = known_smashes[k][selected_dims]

    if victim == 'gradient':
      all_ngrads = np.concatenate(all_grads)

    if victim == 'smash':
      all_smashes = np.concatenate(all_smashes)
      if pre_known:
        for k, item in known_smashes.items():
          known_labels[k] = len(all_smashes)
          all_smashes = np.concatenate([all_smashes, item[np.newaxis, :]])
      all_nsmashes = all_smashes

    all_labels = np.concatenate(all_labels)

    if distance == 'cosine_distance':
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

    # pred_labels = np.ones(len(clusters), dtype='int') * -1
    pred_labels = np.random.choice(labels_set.cpu().numpy().flatten(),
                                   len(clusters))
    for label, index in known_labels.items():
      pred_labels[clusters == clusters[index]] = label

    if pre_known:
      pred_labels = pred_labels[:-len(known_smashes)]

    return pred_labels, all_labels, known_labels, known_smashes
