import torch
import numpy as np
import pdb

def cosineattack_training(pipeline, train_loader, device):
    pred_labels = []
    all_labels = []
    clean_pos_grad = None

    # training mode or eval mode
    pipeline.reset()
    while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.train()

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
                # find all positive labels
                # pdb.set_trace()
                if clean_pos_grad is None:
                  pos_indexes = torch.nonzero(labels == 1, as_tuple=True)[0]
                  # choose one positive grad randomly
                  index = pos_indexes[torch.randperm(len(pos_indexes))][0]
                  clean_pos_grad = outputs[index].reshape(1, -1).clone()
                # calculate the similarities between other grads and the clean positive grad
                grad_cosine = torch.cosine_similarity(outputs.reshape(len(outputs), -1), clean_pos_grad, dim=1) 
                pred_labels.append(grad_cosine.cpu().numpy() > 0.0)
                
                all_labels.append(labels.cpu().numpy())
            inputs = outputs
    return np.concatenate(pred_labels), np.concatenate(all_labels)
