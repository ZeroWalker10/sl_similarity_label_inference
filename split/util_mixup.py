import torch
import datetime
import numpy as np
import pdb

def training_loop(n_epochs, pipeline, train_loader, device, 
                  mixup_alpha=None, in_sae=None):
    # training mode
    pipeline.reset()
    while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.train()

    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        correct = 0
        total = 0
        pipeline.reset()
        while not pipeline.is_end():
            entry = pipeline.next()
            entry.role.on_epoch_start()

        for imgs, labels in train_loader:
            pipeline.reset()
            while not pipeline.is_end():
                entry = pipeline.next()
                entry.role.on_batch_start()
                
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
                    if mixup_alpha is not None:
                        outputs = entry.role.backward(inputs, indexes, lamb)
                    else:
                        outputs = entry.role.backward(inputs)
                else:
                    outputs = entry.role.backward(inputs)
                inputs = outputs


            pipeline.reset()
            while not pipeline.is_end():
                entry = pipeline.next()
                entry.role.scheduler_step(is_batch=True)
                entry.role.on_epoch_end()

            if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
                with torch.no_grad():
                    coae = getattr(pipeline.entries[-1].role, 'coae', None)
                    sae = getattr(pipeline.entries[-1].role, 'sae', None)
                    alter_coaes = getattr(pipeline.entries[-1].role, 'alter_coaes', None)
                    choice = getattr(pipeline.entries[-1].role, 'choice', None)
                    if coae is not None:
                        preds = coae.decoder(predictions)
                        predictions = torch.argmax(preds, dim=1)
                    elif sae is not None:
                        preds = sae.decoder(predictions)
                        predictions = torch.argmax(preds, dim=1)
                    elif alter_coaes is not None:
                        preds = alter_coaes[choice].decoder(predictions)
                        predictions = torch.argmax(preds, dim=1)
                    elif in_sae is not None:
                        preds = in_sae.decoder(predictions)
                        predictions = torch.argmax(preds, dim=1)
                    else:
                        _, predictions = torch.max(predictions, dim=1)
                    total += labels.shape[0]
                    correct += int((predictions == labels).sum())
            
            # release GPU memory
            del imgs, labels, predictions
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            pipeline.reset()
            while not pipeline.is_end():
                entry = pipeline.next()
                entry.role.on_batch_end()

        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            print('{} Epoch {}, Accuracy {:.2f}'.format(
                datetime.datetime.now(), epoch, correct / total
            ))

        # learning rate schedule
        pipeline.reset()
        while not pipeline.is_end():
            entry = pipeline.next()
            entry.role.scheduler_step(is_batch=False)
            entry.role.on_epoch_end()

def training_loop_fake(n_epochs, pipeline, train_loader, device, 
                  mixup_alpha=None, in_sae=None):
    # training mode
    pipeline.reset()
    while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.train()

    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        correct = 0
        total = 0
        pipeline.reset()
        while not pipeline.is_end():
            entry = pipeline.next()
            entry.role.on_epoch_start()

        for imgs, labels, fake_labels in train_loader:
            pipeline.reset()
            while not pipeline.is_end():
                entry = pipeline.next()
                entry.role.on_batch_start()
                
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

            fake_labels = fake_labels.to(device)

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
                    if mixup_alpha is not None:
                        outputs = entry.role.backward(inputs, fake_labels, indexes, lamb)
                    else:
                        outputs = entry.role.backward(inputs, fake_labels)
                else:
                    outputs = entry.role.backward(inputs)
                inputs = outputs


            if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
                with torch.no_grad():
                    coae = getattr(pipeline.entries[-1].role, 'coae', None)
                    sae = getattr(pipeline.entries[-1].role, 'sae', None)
                    alter_coaes = getattr(pipeline.entries[-1].role, 'alter_coaes', None)
                    choice = getattr(pipeline.entries[-1].role, 'choice', None)
                    if coae is not None:
                        preds = coae.decoder(predictions)
                        predictions = torch.argmax(preds, dim=1)
                    elif sae is not None:
                        preds = sae.decoder(predictions)
                        predictions = torch.argmax(preds, dim=1)
                    elif alter_coaes is not None:
                        preds = alter_coaes[choice].decoder(predictions)
                        predictions = torch.argmax(preds, dim=1)
                    elif in_sae is not None:
                        preds = in_sae.decoder(predictions)
                        predictions = torch.argmax(preds, dim=1)
                    else:
                        _, predictions = torch.max(predictions, dim=1)
                    total += labels.shape[0]
                    correct += int((predictions == labels).sum())
            
            # release GPU memory
            del imgs, labels, predictions
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            pipeline.reset()
            while not pipeline.is_end():
                entry = pipeline.next()
                entry.role.on_batch_end()

        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            print('{} Epoch {}, Accuracy {:.2f}'.format(
                datetime.datetime.now(), epoch, correct / total
            ))

        # learning rate schedule
        pipeline.reset()
        while not pipeline.is_end():
            entry = pipeline.next()
            entry.role.scheduler_step()
            entry.role.on_epoch_end()

def validate(pipeline, val_loader, device, in_sae=None, label_mapping=None):
    correct = 0
    total = 0
    with torch.no_grad():
        # eval mode
        pipeline.reset()
        while not pipeline.is_end():
            entry = pipeline.next()
            entry.role.eval()
            
        for imgs, labels in val_loader:
            pipeline.reset()
            inputs = imgs.to(device)
            labels = labels.to(device)
            while not pipeline.is_end():
                entry = pipeline.next()
                outputs = entry.role.forward(inputs)
                inputs = outputs

            coae = getattr(pipeline.entries[-1].role, 'coae', None)
            sae = getattr(pipeline.entries[-1].role, 'sae', None)
            alter_coaes = getattr(pipeline.entries[-1].role, 'alter_coaes', None)
            choice = getattr(pipeline.entries[-1].role, 'choice', None)
            if coae is not None:
                preds = coae.decoder(outputs)
                predictions = torch.argmax(preds, dim=1)
            elif sae is not None:
                preds = sae.decoder(outputs)
                predictions = torch.argmax(preds, dim=1)
            elif alter_coaes is not None:
                preds = alter_coaes[choice].decoder(outputs)
                predictions = torch.argmax(preds, dim=1)
            elif in_sae is not None:
                preds = in_sae.decoder(outputs)
                predictions = torch.argmax(preds, dim=1)
            elif label_mapping is not None:
                predictions = torch.argmax(outputs, dim=1)
                predictions = torch.from_numpy(label_mapping.decode(predictions.cpu().numpy())).to(device)
            else:
                predictions = torch.argmax(outputs, dim=1)
            total += labels.shape[0]
            correct += int((predictions == labels).sum())
    return correct / total

def validate_fake(pipeline, val_loader, device, in_sae=None, label_mapping=None):
    correct = 0
    total = 0
    with torch.no_grad():
        # eval mode
        pipeline.reset()
        while not pipeline.is_end():
            entry = pipeline.next()
            entry.role.eval()
            
        for imgs, labels, fake_labels in val_loader:
            pipeline.reset()
            inputs = imgs.to(device)
            labels = labels.to(device)
            while not pipeline.is_end():
                entry = pipeline.next()
                outputs = entry.role.forward(inputs)
                inputs = outputs

            coae = getattr(pipeline.entries[-1].role, 'coae', None)
            sae = getattr(pipeline.entries[-1].role, 'sae', None)
            alter_coaes = getattr(pipeline.entries[-1].role, 'alter_coaes', None)
            choice = getattr(pipeline.entries[-1].role, 'choice', None)
            if coae is not None:
                preds = coae.decoder(outputs)
                predictions = torch.argmax(preds, dim=1)
            elif sae is not None:
                preds = sae.decoder(outputs)
                predictions = torch.argmax(preds, dim=1)
            elif alter_coaes is not None:
                preds = alter_coaes[choice].decoder(outputs)
                predictions = torch.argmax(preds, dim=1)
            elif in_sae is not None:
                preds = in_sae.decoder(outputs)
                predictions = torch.argmax(preds, dim=1)
            elif label_mapping is not None:
                predictions = torch.argmax(preds, dim=1)
                predictions = torch.from_numpy(label_mapping.decode(predictions.cpu().numpy()))
            else:
                predictions = torch.argmax(outputs, dim=1)
            total += labels.shape[0]
            correct += int((predictions == labels).sum())
    return correct / total

def lm_validate(pipeline, val_loader, device, label_mapping):
    correct = 0
    total = 0
    with torch.no_grad():
        # eval mode
        pipeline.reset()
        while not pipeline.is_end():
            entry = pipeline.next()
            entry.role.eval()
            
        for imgs, labels in val_loader:
            pipeline.reset()
            inputs = imgs.to(device)
            labels = labels.to(device)
            while not pipeline.is_end():
                entry = pipeline.next()
                outputs = entry.role.forward(inputs)
                inputs = outputs

            predictions = label_mapping.decode(outputs.cpu().numpy())
            total += labels.shape[0]
            correct += int((predictions == labels.cpu().numpy()).sum())
    return correct / total

def grads_training(pipeline, train_loader, labels_set, device):
    all_grads = []
    all_labels = []
    known_labels = {}
    offset = 0
    for imgs, labels in train_loader:
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
            outputs = entry.role.backward(inputs)
            if entry.role.step == -1:
                # last step
                for label in labels_set:
                  if label.item() in known_labels:
                    continue
                  label_indexes = torch.nonzero(labels == label,as_tuple=True)[0]
                  if len(label_indexes) == 0:
                      continue
                  # choose one grad randomly
                  index = label_indexes[torch.randperm(len(label_indexes))][0]
                  known_labels[label.item()] = offset + index.item()
                grads = outputs.reshape(len(outputs), -1)
                all_grads.append(grads)
                all_labels.append(labels)
            inputs = outputs
        offset += len(labels)
    return torch.cat(all_grads), torch.cat(all_labels), all_grads, known_labels

def smashed_training(pipeline, train_loader, labels_set, device):
    all_smashed = []
    all_labels = []
    known_labels = {}
    offset = 0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        pipeline.reset()
        inputs = imgs
        # forward
        while not pipeline.is_end():
            entry = pipeline.next()
            outputs = entry.role.forward(inputs)
            inputs = outputs
            if entry.role.step != -1:
                # not last step
                for label in labels_set:
                  if label.item() in known_labels:
                    continue
                  label_indexes = torch.nonzero(labels == label,as_tuple=True)[0]
                  if len(label_indexes) == 0:
                      continue
                  # choose one grad randomly
                  index = label_indexes[torch.randperm(len(label_indexes))][0]
                  known_labels[label.item()] = offset + index.item()
                smashed = outputs.detach().cpu().reshape(len(outputs), -1)
                all_smashed.append(smashed)
                all_labels.append(labels)

        predictions = outputs.to(device)

        # backward
        pipeline.r_reset()
        inputs = labels
        while not pipeline.r_is_end():
            entry = pipeline.r_next()
            outputs = entry.role.backward(inputs)
            inputs = outputs
        offset += len(labels)
    return torch.cat(all_smashed), torch.cat(all_labels), all_smashed, known_labels
