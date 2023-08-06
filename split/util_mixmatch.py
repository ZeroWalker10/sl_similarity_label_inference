import torch
import pdb
import torch.nn.functional as F
import datetime
import numpy as np

def sharpen(p, temp):
    sharpened_p = p ** (1.0 / temp) 
    return sharpened_p / torch.sum(sharpened_p, dim=1).reshape(-1, 1)

def training_loop(n_epochs, pipeline, labeled_train_loader, 
                  unlabeled_train_loader, labels_set, device,
                  mixup_alpha=None, lamb_u=50, temp=0.8, bottom_model_clone=None):
    # training mode
    pipeline.reset()
    while not pipeline.is_end():
        entry = pipeline.next()
        if entry.role.step == -1:
            # update server model
            entry.role.train()

    warm_threshold = n_epochs // 2
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, dummy_labels in unlabeled_train_loader:
            for labeled_features, labeled_labels in labeled_train_loader:
                break
            true_labels = F.one_hot(labeled_labels, num_classes=len(labels_set))
            critical_point = len(true_labels)

            if bottom_model_clone is None or epoch > warm_threshold:
                with torch.no_grad():
                    pipeline.reset()
                    inputs = imgs.to(device)
                    while not pipeline.is_end():
                        entry = pipeline.next()
                        outputs = entry.role.forward(inputs)
                        inputs = outputs

                    outputs = F.softmax(outputs, dim=1)
                    pred_labels = sharpen(outputs, temp).to(imgs.device)
            else:
                unlabeled_intermediate = bottom_model_clone(imgs)
                labeled_intermediate = bottom_model_clone(labeled_features)
                pred_labels = []
                for unlabeled_inter in unlabeled_intermediate:
                    dists = (unlabeled_inter - labeled_intermediate) ** 2
                    shape = list(range(len(dists.shape)))
                    dists = torch.sum(dists, dim=shape[1:])
                    pred_labels.append(labeled_labels[torch.argmin(dists).item()].item())
                pred_labels = torch.from_numpy(np.array(pred_labels))
                pred_labels = F.one_hot(pred_labels, num_classes=len(labels_set))

            imgs = torch.cat([labeled_features, imgs], dim=0)
            labels = torch.cat([true_labels, pred_labels], dim=0)

            if mixup_alpha is not None:
                lamb = np.random.beta(mixup_alpha, mixup_alpha)
                lamb = np.maximum(lamb, 1 - lamb)

                indexes = torch.randperm(imgs.size(0))
                imgs = lamb * imgs + (1 - lamb) * imgs[indexes, :]
                labels = lamb * labels + (1 - lamb) * labels[indexes, :]

                imgs = imgs.to(device)
                labels = labels.to(device)
                indexes = indexes.to(device)
            else:
                imgs = imgs.to(device)
                labels = labels.to(device)

            pipeline.reset()
            inputs = imgs
            # forward
            while not pipeline.is_end():
                entry = pipeline.next()
                outputs = entry.role.forward(inputs)
                inputs = outputs

            predictions = outputs.to(device)

            # backward
            pipeline.r_reset()
            inputs = labels
            while not pipeline.r_is_end():
                entry = pipeline.r_next()
                if entry.role.step == -1:
                    outputs = entry.role.backward(inputs, lamb_u, labels_set, 
                                                  critical_point)
                else:
                    outputs = entry.role.backward(inputs)
                inputs = outputs

            with torch.no_grad():
                _, predictions = torch.max(predictions, dim=1)

            pipeline.reset()
            while not pipeline.is_end():
                entry = pipeline.next()
                entry.role.scheduler_step(is_batch=True)
            
            # release GPU memory
            del imgs, labels, predictions
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # learning rate schedule
        pipeline.reset()
        while not pipeline.is_end():
            entry = pipeline.next()
            entry.role.scheduler_step(is_batch=False)

        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            print('having training {} epochs...'.format(epoch))
