import torch
import copy
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import combinations
import numpy as np
import pdb

eps = 1.0e-35
def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))

def entropy(pred):
    result = -torch.sum(pred * torch.log2(pred + eps))
    return result

def infer_private_label(client_model, server_model, labels_set, 
                        csmashed, cgrads, clabels,
                        clabeled_smashed, clabeled_grads, clabeled_labels,
                        dummy_labels, device, n_epochs=20,
                        lambda1=0.05, lambda2=0.05, lambda3=0.01):
    opt = torch.optim.Adam(list(server_model.parameters()) + [dummy_labels], lr=1)

    for i in range(n_epochs):
        def closure():
            opt.zero_grad()

            for (bsmashed, blabels, bdummy_labels, bgrads) in zip(
                        csmashed, clabels, dummy_labels, cgrads
                    ):
                bsmashed = bsmashed.to(device)
                blabels = blabels.to(device)
                bdummy_labels = bdummy_labels.to(device)
                bgrads = bgrads.to(device)

                dummy_preds = server_model(bsmashed)
                dummy_loss = cross_entropy_for_onehot(dummy_preds, bdummy_labels)
                dummy_grads = torch.autograd.grad(dummy_loss, bsmashed, create_graph=True)

                # gradient distance
                obj = ((dummy_grads[0] - bgrads) ** 2).sum()

                # prediction regularization
                obj = obj + lambda2 * ((dummy_preds - bdummy_labels) ** 2).sum()

            for (labeled_embedding, labeled_labels, labeled_grads) in zip(
                        clabeled_smashed, clabeled_labels, clabeled_grads
                    ):
                labeled_embedding = labeled_embedding.to(device)
                labeled_labels = labeled_labels.to(device)
                labeled_grads = labeled_grads.to(device)

                labeled_preds = server_model(labeled_embedding)
                labeled_loss = cross_entropy_for_onehot(labeled_preds, labeled_labels)
                labeled_gds = torch.autograd.grad(labeled_loss, labeled_embedding,
                                                    create_graph=True)

                obj = obj + lambda1 * ((labeled_gds[0] - labeled_grads[j]) ** 2).sum()
                obj = obj + lambda3 * ((labeled_preds - labeled_labels) ** 2).sum()
                # obj = obj + lambda3 * cross_entropy_for_onehot(labeled_preds, labeled_labs)

            obj.backward()

            return obj
        opt.step(closure)

    return dummy_labels

def learning_based_training(pipeline, train_loader, labels_set, device, 
                            labeled_loader, loss_fn, surrogate_top, strong=False, n_epoch=100,
                            lambda1=0.05, lambda2=0.05, lambda3=0.01, lambda4=0.05, k=0):
    # training mode or eval mode
    pipeline.reset()
    while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.train()

    client_model = copy.deepcopy(pipeline.entries[0].role.sub_model)
    server_model = copy.deepcopy(pipeline.entries[-1].role.sub_model)
    surrogate_model = surrogate_top 

    clabeled_labels = []
    cunlabeled_labels = []
    dummy_labels = []
    for i, loader in enumerate([labeled_loader, train_loader]):
        labels = []
        for data, labs in loader:
            if i == 0:
                labs_onehot = F.one_hot(labs, len(labels_set)).cpu()
                labels.append(labs_onehot)

            if i == 1:
                cunlabeled_labels.append(labs)
                labs_onehot = F.one_hot(labs, len(labels_set)).cpu().float()
                dummy_labels.append(
                    torch.randn_like(labs_onehot).to(device).requires_grad_(True)
                )

        if i == 0:
            clabeled_labels = labels

    clabeled_labels = torch.cat(clabeled_labels)
    cunlabeled_labels = torch.cat(cunlabeled_labels)
    dummy_clone = copy.deepcopy(dummy_labels)
    surrogate_model_clone = copy.deepcopy(surrogate_model)
    opt = torch.optim.Adam(list(surrogate_model.parameters()) + dummy_labels, lr=1e-2)
    for i in range(n_epoch):
        offset = 0
        for j, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels_onehot = F.one_hot(labels, len(labels_set)).to(device)

            embedding = client_model(imgs)
            predictions = server_model(embedding)
            # loss = loss_fn(predictions, labels)
            loss = cross_entropy_for_onehot(predictions, labels_onehot)
            gds = torch.autograd.grad(loss, embedding, create_graph=True)[0].detach()

            def closure():
                agg_loss = 0.0
                agg_grad_diff = 0.0

                opt.zero_grad()
                # labels_onehot = F.one_hot(labels, len(labels_set)).cpu()
                # dlabels = dummy_labels[offset:offset+len(embedding)]
                dlabels = dummy_labels[j]

                dpreds = surrogate_model(embedding)
                dloss = cross_entropy_for_onehot(dpreds, dlabels)
                dgds = torch.autograd.grad(dloss, embedding, create_graph=True)[0]
                grad_diff = ((dgds - gds) ** 2).sum()
                agg_grad_diff = agg_grad_diff + grad_diff
                agg_loss = agg_loss + grad_diff
                # agg_loss = agg_loss + lambda2 * ((dpreds - dlabels) ** 2).sum()
                agg_loss = agg_loss + lambda2 * cross_entropy_for_onehot(dpreds, F.softmax(dlabels, 1))
                # known
                for labeled_data, labeled_labels in labeled_loader:
                    labeled_data = labeled_data.to(device)
                    labeled_labels = F.one_hot(labeled_labels, len(labels_set)).to(device)

                    labeled_embedding = client_model(labeled_data)
                    labeled_dpreds = surrogate_model(labeled_embedding)
                    # labeled_dloss = loss_fn(labeled_dpreds, labeled_labels)
                    labeled_dloss = cross_entropy_for_onehot(labeled_dpreds, labeled_labels)
                    labeled_dgds = torch.autograd.grad(labeled_dloss, labeled_embedding,
                                                      create_graph=True)[0]

                    labeled_preds = server_model(labeled_embedding)
                    # labeled_loss = loss_fn(labeled_preds, labeled_labels)
                    labeled_loss = cross_entropy_for_onehot(labeled_preds, labeled_labels)
                    labeled_gds = torch.autograd.grad(labeled_loss, labeled_embedding,
                                                      create_graph=True)[0]

                    grad_diff = ((labeled_dgds - labeled_gds) ** 2).sum()
                    agg_grad_diff = agg_grad_diff + grad_diff
                    agg_loss = agg_loss + lambda1 * grad_diff
                    # agg_loss = agg_loss + lambda3 * loss_fn(labeled_dpreds, labeled_labels) 
                    agg_loss = agg_loss + lambda3 * cross_entropy_for_onehot(labeled_dpreds, labeled_labels) 

                # dlabels.retain_grad()
                agg_loss.backward()
                # return agg_grad_diff
                return agg_loss

            opt.step(closure)
            offset = offset + len(embedding)
        
    dummy_labels = torch.cat(dummy_labels).detach()
    pred_labels = torch.argmax(dummy_labels, dim=1).cpu()
    return pred_labels, cunlabeled_labels

def save_learning_based_training(pipeline, train_loader, labels_set, device, 
                            labeled_loader, loss_fn, surrogate_top, strong=False, n_epoch=100,
                            lambda1=0.05, lambda2=0.05, lambda3=0.01, lambda4=0.05, k=0):
    # training mode or eval mode
    pipeline.reset()
    while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.train()

    client_model = copy.deepcopy(pipeline.entries[0].role.sub_model)
    server_model = copy.deepcopy(pipeline.entries[-1].role.sub_model)
    surrogate_model = surrogate_top 

    clabeled_labels = []
    clabeled_gds = []
    # clabeled_data = []
    clabeled_embeddings = []
    dummy_labels = []
    cunlabeled_labels = []
    cunlabeled_gds = []
    # cunlabeled_data = []
    cunlabeled_embeddings = []
    org_labels = []
    for i, loader in enumerate([labeled_loader, train_loader]):
        labels = []
        for data, labs in loader:
            if i == 0:
                labs_onehot = F.one_hot(labs, len(labels_set)).cpu()
                clabeled_labels.append(labs_onehot)

                embedding = client_model(data.to(device))
                predictions = server_model(embedding)
                loss = cross_entropy_for_onehot(predictions, labs_onehot.to(device))
                gds = torch.autograd.grad(loss, embedding, create_graph=True)[0].detach().cpu()
                clabeled_gds.append(gds)
                # clabeled_data.append(data)
                clabeled_embeddings.append(embedding.detach().cpu())
            if i == 1:
                org_labels.append(labs)
                labs_onehot = F.one_hot(labs, len(labels_set)).cpu().float()
                dummy_labels.append(
                    torch.randn_like(labs_onehot).to(device).requires_grad_(True)
                )
                cunlabeled_labels.append(labs_onehot)

                embedding = client_model(data.to(device))
                predictions = server_model(embedding)
                loss = cross_entropy_for_onehot(predictions, labs_onehot.to(device))
                gds = torch.autograd.grad(loss, embedding, create_graph=True)[0].detach().cpu()
                cunlabeled_gds.append(gds)
                # cunlabeled_data.append(data)
                cunlabeled_embeddings.append(embedding.detach().cpu())

    org_labels = torch.cat(org_labels)
    opt = torch.optim.Adam(list(surrogate_model.parameters()) + dummy_labels, lr=1e-2)
    for i in range(n_epoch):
        def closure():
            opt.zero_grad()
            agg_loss = 0

            '''
            for (data, score, dummy_score, org_gds, embeddings) in zip(cunlabeled_data, cunlabeled_labels,
                                                            dummy_labels, cunlabeled_gds,
                                                            cunlabeled_embeddings):
            '''
            for (score, dummy_score, org_gds, embeddings) in zip(cunlabeled_labels,
                                                            dummy_labels, cunlabeled_gds,
                                                            cunlabeled_embeddings):
                # embeddings = client_model(data.to(device))
                embeddings = embeddings.to(device)
                preds = surrogate_model(embeddings.requires_grad_(True))
                dloss = cross_entropy_for_onehot(preds, score.to(device))
                dgds = torch.autograd.grad(dloss, embeddings, create_graph=True, allow_unused=True)[0]
                grad_diff = ((dgds - org_gds.to(device)) ** 2).sum()

                grad_diff = grad_diff + lambda2 * cross_entropy_for_onehot(preds, F.softmax(dummy_score, 1))

                agg_loss = agg_loss + grad_diff

                '''
                for (data, score, org_gds, embeddings) in zip(clabeled_data, clabeled_labels, clabeled_gds,
                                                    clabeled_embeddings):
                '''
                for (score, org_gds, embeddings) in zip(clabeled_labels, clabeled_gds,
                                                    clabeled_embeddings):
                    # embeddings = client_model(data.to(device))
                    embeddings = embeddings.to(device)
                    preds = surrogate_model(embeddings.requires_grad_(True))
                    dloss = cross_entropy_for_onehot(preds, score.to(device))
                    dgds = torch.autograd.grad(dloss, embeddings, create_graph=True, allow_unused=True)[0]

                    grad_diff = ((dgds - org_gds.to(device)) ** 2).sum()
                    agg_loss = agg_loss + lambda1 * grad_diff
                    agg_loss = agg_loss + lambda3 * cross_entropy_for_onehot(preds, score.to(device)) 

            agg_loss.backward()
            return agg_loss
        opt.step(closure)
        
    dummy_labels = torch.cat(dummy_labels).detach()
    pred_labels = torch.argmax(dummy_labels, dim=1).cpu()
    return pred_labels, org_labels
