#encoding=utf8

import os
import json
import itertools
import numpy as np
from tqdm import tqdm
from sklearn.metrics import pairwise_distances, accuracy_score
from .evaluation import result_eval

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

    def transform(self, embeds):
        embeds = (embeds + self.bias).dot(self.kernel)
        norms = (embeds**2).sum(axis=1, keepdims=True)**0.5
        embeds = embeds / np.clip(norms, 1e-8, np.inf)
        return embeds

class BasicDatastore(object):
    def __init__(self):
        self.embeds = None
        self.probs = None
        self.dataset_name = None
        self.dataset_path = None
        self.embed_transformer = WhiteningTransformer()
        self.data_enhancer = PrototypeEnhancer()
    
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

    def transform_embed(self, embeds):
        return self.embed_transformer.transform(embeds)

    def construct(self, classify_model, data_model, dataset_name):
        """ 建立需要索引的datastore
        dataset_name: 数据集名字
        """
        if dataset_name == "unlabeled":
            data_loader = data_model.unlabeled_dataloader()
        else:
            data_loader = data_model.knn_datastore_dataloader()
        embeds = []
        probs = []
        label_embeds_dict = {} # 每个label下所有样本的向量表示

        for batch in tqdm(data_loader):
        # for batch in data_loader:
            logits, inner_probs, _, labels, sample_embeds = classify_model.predict(batch)
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
        self.embed_transformer.preprocess(np.vstack(embeds))
        label_enhanced_embeds_dict = {}
        if self.data_enhancer:
            label_enhanced_embeds_dict = self.data_enhancer.enhance(label_embeds_dict)
        # 在样本空间中采样增强datastore
        for label, enhanced_embeds in label_enhanced_embeds_dict.items():
            enhanced_embeds = np.vstack(enhanced_embeds)
            embeds.append(enhanced_embeds)
            probs.append(np.eye(inner_probs.shape[1])[[label] * enhanced_embeds.shape[0]])
        embeds = np.vstack(embeds)
        probs = np.vstack(probs)
        
        embeds = self.embed_transformer.transform(embeds)
        self.embeds = embeds
        self.probs = probs
        self.dataset_name = dataset_name
        datastore_dict = {
            "embeds": embeds,
            "probs": probs,
            "dataset_name": dataset_name,
            "embed_transformer": self.embed_transformer,
        }

        print("datastore[%s] embed size:" % dataset_name, embeds.shape)
        return datastore_dict

def get_datastores(dataset_names, model, data_model):
    """ 加载or计算索引 """
    datastores = []
    for dataset_name in dataset_names:
        datastore = BasicDatastore()
        datastore.construct(
            classify_model=model,
            data_model=data_model,
            dataset_name=dataset_name
        )
        datastores.append(datastore)
    return datastores

def get_classify_info(model, data_loader):
    """ 计算分类的概率 """
    y_true = []
    y_pred = []
    classify_probs = []
    sample_embeds = []

    for batch in tqdm(data_loader):
    # for batch in data_loader:
        _, pred_probs, _, labels, embeds = model.predict(batch)
        y_true += list(labels)
        sample_embeds.append(embeds)
        classify_probs.append(pred_probs)
    
    classify_probs = np.vstack(classify_probs)
    sample_embeds = np.vstack(sample_embeds)
    y_pred = list(np.argmax(classify_probs, axis=1))

    print("model classify prob shape:", classify_probs.shape)
    print("sample embed original shape:", sample_embeds.shape)

    return y_true, y_pred, classify_probs, sample_embeds

def softmax_2D(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return softmax_x

def get_knn_probs(distances, datastore_probs, k, remove_self=False):
    """计算knn的概率

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

def grid_search_for_hyper(model, data_model, datastores):
    """ 超参搜索 """
    y_true, _, classify_probs, sample_embeds = get_classify_info(model, data_model.val_dataloader())

    best_acc = 0.0
    best_hyper = {"lambda_values": [], "k_values": []}
    best_y_pred = []
    
    # 先算好distances
    distance_cache = []
    for datastore in datastores:
        query_embeds = datastore.transform_embed(sample_embeds)
        datastore_embeds = datastore.get_datastore_embed()
        distances = pairwise_distances(query_embeds, datastore_embeds, metric="euclidean", n_jobs=1)
        distance_cache.append(distances)
    knn_cache = []
    for index in range(len(datastores)):
        knn_cache.append({})

    k_search_list = [np.arange(0, 32, 1).tolist()] * len(datastores)
    lambda_search_list = [np.linspace(0, 1, 21).tolist()] * len(datastores)
    for hyper_tuple in itertools.product(*lambda_search_list, *k_search_list):
        lambda_values = list(map(lambda x: round(x, 2), hyper_tuple[:len(lambda_search_list)]))
        k_values = list(hyper_tuple[len(lambda_search_list):])
        # print("start for search", lambda_values, k_values)
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
                knn_probs = get_knn_probs(distance_cache[index], datastore_probs, k_values[index])
                knn_cache[index][k_values[index]] = knn_probs
            final_prob = final_prob + lambda_values[index] * knn_probs
        y_pred = list(np.argmax(final_prob, axis=1))
        current_acc = accuracy_score(y_true, y_pred)
        # print("current_acc", current_acc)
        if current_acc > best_acc:
            best_acc = current_acc
            best_y_pred = y_pred
            best_hyper = {"lambda_values": lambda_values, "k_values": k_values}
            print("best acc:", round(best_acc * 100, 2), "\tbest hyper:", best_hyper)
    return best_hyper, y_true, best_y_pred

def knn_augmentation(model, data_model, output_dir):
    """ 使用kNN增加预测的效果
    Args:
        model: 模型对象，BaseModel的子类
        data_model: 数据模型的对象，TaskDataModel的子类
        output_dir: 需要输出到的目录
    """
    model.cuda()
    model.eval()
    data_model.setup("test")
    datastore_names = ["labeled"]
    datastores = get_datastores(datastore_names, model, data_model)
    best_hyper, dev_y_true, dev_best_y_pred = grid_search_for_hyper(model, data_model, datastores)
    return best_hyper, datastores

def knn_inference(sample_embeds, knn_datastores, knn_best_hyper):
    total_knn_probs = None
    for l, k, datastore in zip(knn_best_hyper["lambda_values"], knn_best_hyper["k_values"], knn_datastores):
        if l <= 1e-5 or k <= 0:
            continue
        remove_self = True if datastore.get_dataset_name() == "unlabeled" else False # 检索时是否移除自身
        if remove_self:
            print("remove itself, leave one out.")
            k = k + 1
        query_embeds = datastore.transform_embed(sample_embeds)
        datastore_embeds = datastore.get_datastore_embed()
        datastore_probs = datastore.get_datastore_prob()
        distances = pairwise_distances(query_embeds, datastore_embeds, metric="euclidean", n_jobs=1)
        knn_probs = get_knn_probs(
            distances, datastore_probs, k, remove_self=remove_self)
        if total_knn_probs is None:
            total_knn_probs = knn_probs
        else:
            total_knn_probs = total_knn_probs + l * knn_probs
    return total_knn_probs
