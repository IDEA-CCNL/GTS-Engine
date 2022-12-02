#encoding=utf8

import os
import json
import itertools
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import pairwise_distances, accuracy_score

class PrototypeEnhancer(object):
    def __init__(self):
        pass

    def enhance(self, label_embeds_dict):
        label_enhanced_embeds_dict = {}
        enhance_num = 0
        for label, label_embeds in label_embeds_dict.items():
            label_embeds = np.vstack(label_embeds)
            embed_size = label_embeds.shape[0]
            label_enhanced_embeds_dict[label] = [np.mean(label_embeds, axis=0)]
            enhance_num += 1
            for c_embeds in itertools.combinations(label_embeds, embed_size - 1):
                if len(c_embeds) <= 0:
                    continue
                c_embeds = np.vstack(c_embeds)
                label_enhanced_embeds_dict[label].append(np.mean(c_embeds, axis=0))
                enhance_num += 1
        print("enhance num:", enhance_num)
        return label_enhanced_embeds_dict

class WhiteningTransformer(object):
    def __init__(self):
        self.kernel = None
        self.bias = None

    def preprocess(self, embeds, n_components=64):
        """计算whitening的kernel和bias
        vecs.shape = [num_samples, embedding_size]，
        最后的变换：y = (x + bias).dot(kernel)
        """
        mu = embeds.mean(axis=0, keepdims=True)
        cov = np.cov(embeds.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        if n_components > 0:
            W = W[:, :n_components]
        self.kernel = W
        self.bias = -mu
        return self.kernel ,self.bias

    def transform_raw(self, embeds):
        embeds = (embeds + self.bias).dot(self.kernel)
        norms = (embeds**2).sum(axis=1, keepdims=True)**0.5
        embeds = embeds / np.clip(norms, 1e-8, np.inf)
        return embeds

    def transform(self, embeds, kernel, bias):
        embeds = torch.matmul(embeds + bias, kernel)  # type: ignore
        norms = (embeds**2).sum(dim=1, keepdims=True)**0.5  # type: ignore
        embeds = embeds / torch.clamp(norms, 1e-8, np.inf)
        return embeds

class BasicDatastore(object):
    def __init__(self, n_components=64):
        self.embeds = None
        self.probs = None
        self.dataset_name = None
        self.dataset_path = None
        self.embed_transformer = WhiteningTransformer()
        self.data_enhancer = PrototypeEnhancer()
        self.n_components = n_components
        self.kernel = None
        self.bias = None
    
    def get_embed_transformer(self):
        return self.embed_transformer

    def get_datastore_embed(self):
        return self.embeds

    def get_datastore_prob(self):
        return self.probs

    def get_dataset_name(self):
        return self.dataset_name

    def get_dataset_path(self):
        return self.dataset_path

    def get_kernel(self):
        return self.kernel

    def get_bias(self):
        return self.bias

    def transform_embed(self, embeds):
        return self.embed_transformer.transform(embeds, self.kernel, self.bias)
    
    def transform_embed_raw(self, embeds):
        return self.embed_transformer.transform_raw(embeds)

    def construct(self, classify_model, dataset, dataset_name="labeled"):
        """ 建立需要索引的datastore
        """
        classify_model.eval()
        embeds = []
        probs = []
        device = None
        label_embeds_dict = {} # 每个label下所有样本的向量表示
        for batch in dataset:
            labels = batch['label_id_clf']
            inference_output = classify_model(input_ids=batch['input_ids'],
                    input_mask=batch['input_mask'],
                    input_seg=batch['input_seg'],
                    prompt_gate=False)
            device = inference_output['positions'].device
            sample_embeds = np.array(inference_output['embeds'].detach().cpu())
            inner_probs = np.array(inference_output['probs'].detach().cpu())
            if dataset_name != "unlabeled":
                hard_probs = np.eye(inner_probs.shape[1])[labels]
                batch_probs = hard_probs
                probs.append(batch_probs)
            else:
                probs.append(inner_probs)
            embeds.append(sample_embeds)
            if dataset_name != "unlabeled":
                for label, embed in zip(labels, sample_embeds):
                    if label not in label_embeds_dict:
                        label_embeds_dict[label] = []
                    label_embeds_dict[label].append(embed)   
        # 构建句子向量变换器
        kernel ,bias = self.embed_transformer.preprocess(np.vstack(embeds), self.n_components)
        # 在样本空间中采样增强datastore
        label_enhanced_embeds_dict = {}
        if self.data_enhancer:
            label_enhanced_embeds_dict = self.data_enhancer.enhance(label_embeds_dict)
        for label, enhanced_embeds in label_enhanced_embeds_dict.items():
            enhanced_embeds = np.vstack(enhanced_embeds)
            embeds.append(enhanced_embeds)
            indices = label.detach().cpu().tolist()* enhanced_embeds.shape[0]
            probs.append(np.eye(inner_probs.shape[1])[indices])  # type: ignore
        embeds = np.vstack(embeds)
        probs = np.vstack(probs)

        self.kernel = torch.tensor(kernel).float().to(device=device)
        self.bias = torch.tensor(bias).float().to(device=device)
        embeds = self.embed_transformer.transform(torch.tensor(embeds).float().to(device=device), self.kernel, self.bias)
        self.embeds = embeds
        self.probs = torch.tensor(probs).float().to(device=device)
        self.dataset_name = dataset_name
        
        datastore_dict = {
            "embeds": self.embeds,
            "probs": self.probs,
            "dataset_name": self.dataset_name,
            "kernel": self.kernel,
            "bias": self.bias,
            "embed_transformer": self.embed_transformer,
        }

        print("datastore[%s] embed size:" % dataset_name, embeds.shape)
        return datastore_dict

def get_datastore(model, data_model):
    """ 加载or计算索引 """
    
    datastore = BasicDatastore()
    datastore.construct(
        classify_model=model,
        dataset=data_model  # type: ignore
    )
    return datastore

def softmax_2D(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return softmax_x

def get_knn_probs_raw(distances, datastore_probs, k, remove_self=False):
    """numpy算子，计算knn的概率

    Args:
        distances: pair-wise距离矩阵
        datastore_probs: 索引对应的样本概率分布
        k: knn的k值
        remove_self (bool, optional): 计算knn时是否移除自身，当query样本和datastore样本来源相同时，需要设置为True. Defaults to False.
    """
    nearest_indices = np.argpartition(distances, kth=k, axis=1)[:, :k] # 最近的k个下标
    nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1) # 最近的k个距离
    if remove_self:
        valid_distances = (nearest_distances >= 1e-5)
        # 有可能存在多个距离等于0的情况，只保留一个
        invalid_row_indices = np.where(np.count_nonzero(valid_distances, axis=1) != k-1)[0] # 存在问题的行
        for invalid_row_index in invalid_row_indices:
            invalid_col_indices = np.where(valid_distances[invalid_row_index] == False)[0] # 存在问题的列
            for invalid_col_index in invalid_col_indices[1:]:
                valid_distances[invalid_row_index][invalid_col_index] = True # 只保留一个
        nearest_indices = nearest_indices[valid_distances].reshape(distances.shape[0], k-1)
        nearest_distances = nearest_distances[valid_distances].reshape(distances.shape[0], k-1)
    nearest_distances = softmax_2D(-1.0 * nearest_distances) # norm distance
    # nearest_distances = 1.0 - softmax_2D(nearest_distances)
    nearest_prob = np.take(datastore_probs, nearest_indices, axis=0)
    nearest_prob = np.transpose(nearest_prob, (0, 2, 1))
    nearest_distances = np.expand_dims(nearest_distances, axis=2)
    nearest_distances = np.repeat(nearest_distances, datastore_probs.shape[-1], axis=2)
    nearest_distances = np.transpose(nearest_distances, (0, 2, 1))
    nearest_prob = nearest_prob * nearest_distances
    nearest_prob = np.sum(nearest_prob, axis=2)
    return nearest_prob

def get_knn_probs(distances, datastore_probs, k):
    """torch算子，计算knn的概率

    Args:
        distances: pair-wise距离矩阵
        datastore_probs: 索引对应的样本概率分布
        k: knn的k值
    """
    nearest_indices = torch.topk(distances, k=k, dim=1, largest=False).indices # 最近的k个下标
    nearest_distances = torch.gather(distances, 1, nearest_indices) # 最近的k个距离
    m = torch.nn.Softmax(1)
    nearest_distances = m(-1.0*nearest_distances)
    nearest_prob = torch.gather(datastore_probs.unsqueeze(0).repeat(nearest_indices.shape[0],1,1), 1, \
                                        nearest_indices.unsqueeze(-1).repeat(1,1,datastore_probs.shape[-1]))
    nearest_prob = nearest_prob.permute(0,2,1)
    nearest_distances = nearest_distances.unsqueeze(2).repeat(1,1,datastore_probs.shape[-1]).permute(0,2,1)
    nearest_prob = nearest_prob * nearest_distances
    nearest_prob = nearest_prob.sum(2)
    return nearest_prob

def grid_search_for_hyper(datastores, dev_dataloader, classify_model):
    classify_model.eval()
    datastores= [datastores]
    """ 加载验证数据 """
    y_true = []
    classify_probs = []
    sample_embeds = []
    for batch in dev_dataloader:
        labels = batch['label_id_clf']
        inference_output = classify_model(input_ids=batch['input_ids'],
                input_mask=batch['input_mask'],
                input_seg=batch['input_seg'],
                prompt_gate=False)
        embeds = np.array(inference_output['embeds'].detach().cpu())
        inner_probs = np.array(inference_output['probs'].detach().cpu())
        y_true.extend(labels)
        classify_probs.append(inner_probs)
        sample_embeds.append(embeds)
    sample_embeds = np.vstack(sample_embeds)
    classify_probs = np.vstack(classify_probs)

    """ 超参搜索 """
    best_acc = 0.0
    best_hyper = {"lambda_values": [], "k_values": []}
    best_y_pred = []
    
    # compute distances
    distance_cache = []
    for datastore in datastores:
        query_embeds = datastore.transform_embed_raw(sample_embeds)
        datastore_embeds = datastore.get_datastore_embed()
        distances = pairwise_distances(query_embeds, datastore_embeds.detach().cpu().numpy(), metric="euclidean", n_jobs=1)
        distance_cache.append(distances)

    # search for best hyper parameters
    knn_cache = []
    for index in range(len(datastores)):
        knn_cache.append({})
    k_search_list = [np.arange(0, 32, 1)]
    lambda_search_list = [np.linspace(0, 1, 21)]
    for hyper_tuple in itertools.product(*lambda_search_list, *k_search_list):
        lambda_values = list(map(lambda x: round(x, 2), hyper_tuple[:len(lambda_search_list)]))
        k_values = list(hyper_tuple[len(lambda_search_list):])
        y_pred = []
        if sum(lambda_values) > 1:
            continue
        final_prob = (1 - sum(lambda_values)) * classify_probs
        for index in range(len(datastores)):
            datastore = datastores[index]
            if k_values[index] == 0 or lambda_values[index] <= 1e-4:
                continue
            if k_values[index] in knn_cache[index]:
                knn_probs = knn_cache[index][k_values[index]]
            else:
                datastore_probs = datastore.get_datastore_prob()
                knn_probs = get_knn_probs_raw(distance_cache[index], datastore_probs.detach().cpu().numpy(), k_values[index])
                knn_cache[index][k_values[index]] = knn_probs
            final_prob = final_prob + lambda_values[index] * knn_probs
        y_pred = list(np.argmax(final_prob, axis=1))
        current_acc = accuracy_score(y_true, y_pred)
        if current_acc > best_acc:
            best_acc = current_acc
            best_y_pred = y_pred
            best_hyper = {"lambda_values": lambda_values[0], "k_values": k_values[0]}
            print("best acc:", round(best_acc * 100, 2), "\tbest hyper:", best_hyper)

    return best_hyper, y_true, best_y_pred
    
def pdist(a: torch.Tensor, b: torch.Tensor, p: int = 2) -> torch.Tensor:
    a=a.unsqueeze(1)
    b=b.unsqueeze(0)
    return (a-b).abs().pow(p).sum(-1).pow(1/p)

def inference_with_knn(datastore, classify_probs, sample_embeds, best_hyper, whitening=True):
    datastore_embeds = datastore.get_datastore_embed()
    sample_embeds = sample_embeds.to(datastore_embeds.device)
    classify_probs = classify_probs.to(datastore_embeds.device)
    # embedding预处理
    query_embeds = datastore.transform_embed(sample_embeds)
    # 先算好distances
    distances = pdist(query_embeds, datastore_embeds)
    lambda_values = best_hyper["lambda_values"]
    k_values = best_hyper["k_values"]
    if k_values == 0 or lambda_values <= 1e-4:
        return classify_probs
    else:
        final_prob = (1 - lambda_values) * classify_probs
        datastore_probs = datastore.get_datastore_prob()
        knn_probs = get_knn_probs(distances, datastore_probs, k_values)
        final_prob = final_prob + lambda_values * knn_probs
    return final_prob
