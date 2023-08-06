import torch

def normattack_training(pipeline, train_loader, device):
    grad_norms = []
    all_labels = []
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
                grad_norm = outputs.reshape(len(outputs), -1).pow(2).sum(dim=1).sqrt()
                grad_norms.append(grad_norm)
                all_labels.append(labels)
            inputs = outputs
    return torch.cat(grad_norms), torch.cat(all_labels)
