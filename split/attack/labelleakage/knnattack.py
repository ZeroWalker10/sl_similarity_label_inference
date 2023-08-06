import torch
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import preprocessing
import pdb
import numpy as np

def select_features(orig_features, max_dims):
  vt = VarianceThreshold()
  vt.fit(X=orig_features)
  selected_dims = np.sort(np.argsort(vt.variances_)[-max_dims:])
  return selected_dims

def knnattack_training(pipeline, train_loader, labels_set, device, pca_components=-1, distance='euclidean', victim='gradient',
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
    
    batch_size = None
    max_features = 10240
    max_samples = len(labels_set) * 5
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
                    if selected is None:
                      selected_candidates = batch_smashes
                    else:
                      selected_candidates = np.concatenate([selected_candidates, batch_smashes])
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
            outputs = entry.role.backward(inputs)
            if entry.role.step == -1:
                # last step
                # pdb.set_trace()
                if victim == 'gradient':
                    batch_grads = outputs.cpu().numpy().reshape(len(imgs), -1)
                    if batch_grads.shape[1] > max_features:
                      if selected_dims is None:
                        if selected_candidates is None:
                          selected_candidates = batch_grads
                        else:
                          selected_candidates = np.concatenate([selected_candidates, batch_grads.tolist()])
                      else:
                        all_grads.append(batch_grads[:, selected_dims].tolist())
                    else:
                      all_grads.append(batch_grads)
            inputs = outputs

        if selected_candidates is not None and \
            len(selected_candidates) >= max_samples and \
          selected_dims is None:
          selected_dims = select_features(selected_candidates,
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
        if len(known_smashes[k]) > selected_dims:
          known_smashes[k] = known_smashes[selected_dims]

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

    pred_dists = np.ones((len(all_ngrads), len(labels_set)), dtype='float') * np.inf
    for label, index in known_labels.items():
      pred_dists[:, label] = np.linalg.norm(all_ngrads - all_ngrads[index], axis=1)
      pred_labels = np.argmin(pred_dists, axis=1)

    return pred_labels, all_labels, known_labels, known_smashes
