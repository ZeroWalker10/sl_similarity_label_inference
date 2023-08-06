import torch
import numpy as np
import pdb
import copy

def unsplitattack_training(pipeline, train_loader, labels_set, device, strong=False):
    pred_labels = []
    all_labels = []

    # training mode or eval mode
    pipeline.reset()
    while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.train()


    if not strong:
        last_entry_clone = pipeline.entries[-1].clone()
    else:
        last_entry_clone = copy.deepcopy(pipeline.entries[-1])
    # last_entry_clone.role.sub_model = last_entry_clone.role.sub_model.cpu()
    last_entry_clone.role.sub_model = last_entry_clone.role.sub_model

    measure_fn = torch.nn.MSELoss()
    for imgs, labels in train_loader:
        repeat_shape = (1,) * len(imgs.shape[1:])
        imgs = imgs.repeat((2,) + repeat_shape)
        labels = labels.repeat((2,))

        imgs = imgs.to(device)
        labels = labels.to(device)
        
        pipeline.reset()
        inputs = imgs

        pred_outputs = None
        target_grads = None
        # forward
        while not pipeline.is_end():
            entry = pipeline.next()
            outputs = entry.role.forward(inputs)
            if entry.role.step == -1:
                # last step
                clone_outputs = last_entry_clone.role.forward(inputs)
            inputs = outputs
        
        predictions = outputs.to(device)
        
        # backward
        pipeline.r_reset()
        inputs = labels
        pred_label = None
        while not pipeline.r_is_end():
            entry = pipeline.r_next()
            outputs = entry.role.backward(inputs)
            if entry.role.step == -1:
                # last step
                target_grad = [param.grad.detach() for param in entry.role.sub_model.parameters()][:1]
                clone_losses = [last_entry_clone.role.loss_fn(clone_outputs, label_candidate.repeat((2,)).to(device))
                                for label_candidate in labels_set]

                index, min_delta = -1, -1 
                for ind, loss_candidate in enumerate(clone_losses):
                    clone_grad = torch.autograd.grad(loss_candidate, last_entry_clone.role.sub_model.parameters(),
                            allow_unused=True, retain_graph=True)[:1]
                    delta_grad = torch.sum(torch.Tensor(list(measure_fn(cgd, tgd)
                                                             for (cgd, tgd) in zip(clone_grad, target_grad))))
                    if min_delta < 0 or (delta_grad.item() < min_delta.item()):
                        min_delta = delta_grad
                        index = ind 

                # clone_grads = [torch.autograd.grad(loss_candidate, last_entry_clone.role.sub_model.parameters(),
                #                                   allow_unused=True, retain_graph=True)
                #               for loss_candidate in clone_losses]
                # delta_grads = [torch.sum(torch.Tensor(list(measure_fn(cgd, tgd)
                #                for (cgd, tgd) in zip(clone_grad, target_grad))))
                #               for clone_grad in clone_grads]
                # pred_label = torch.argmin(torch.Tensor(delta_grads)).unsqueeze(0)
                # pred_labels.append(pred_label.cpu().numpy())
                pred_label = torch.Tensor([index]).long().to(device)
                pred_labels.append(np.array([index]))
                all_labels.append(labels[0].unsqueeze(0).cpu().numpy())
            inputs = outputs
        
        # update the clone layer
        last_entry_clone.role.backward(pred_label.repeat((2,)))
    return np.concatenate(pred_labels), np.concatenate(all_labels)
