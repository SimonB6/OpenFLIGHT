from abc import ABC, abstractmethod
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer, Client_Init

import torch
import os
import numpy as np
import logging
import math

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from collections import Counter
import math

logger = logging.getLogger(__name__)

# def vectorize_net(net):
#     return torch.cat([p.view(-1) for p in net.parameters()])



def vectorize_net_dict(net):
    return torch.cat([net[key].view(-1) for key in net])


# def load_model_weight(net, weight):
#     index_bias = 0
#     for p_index, p in enumerate(net.parameters()):
#         p.data =  weight[index_bias:index_bias+p.numel()].view(p.size())
#         index_bias += p.numel()


def load_model_weight_dict(net, weight):
    index_bias = 0
    for p_index, p in net.items():
        net[p_index].data = weight[index_bias:index_bias + p.numel()].view(
            p.size())
        index_bias += p.numel()

def direction(tensor1, tensor2):
    dot_product = torch.dot(tensor1.view(-1), tensor2.view(-1))
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)
    cosine_sim = dot_product / (norm1 * norm2)
    return cosine_sim.item()

def regular(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def regular_axis(x):
    x = np.array(x)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)

def dbscan_clustering(data, eps=0.5, min_samples=2, metric='precomputed'):
    data = np.array(data)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(data)
    dbscan.fit(data)
    labels = dbscan.labels_
    return labels

def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))

class Aggregator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def aggregate(self, agg_info):
        pass


class ClientsAvgAggregator(Aggregator):
    def __init__(self, model=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config
        self.general_total_time = 0
        self.total_time = [0] * 105
        self.pre_value = [None] * 105
        self.pri_benign = [None] * 105
        self.first_flag = True
        self.benign_model = None
        self.cluster_benign_model = None

        self.final_model = [None] * 105
        self.gussian_model = [None] * 105
    
    def allow_benign_model_init(self, models):
        for i in range(len(models)):
            client_id, local_sample_size, local_model = models[i]
            if i == 0 and self.first_flag == True:
                self.benign_model = local_model
            if self.pre_value[client_id] == None:
                self.pre_value[client_id] = local_model
            if client_id in Client_Init:
                self.pre_value[client_id] = self.benign_model
            self.pri_benign[client_id] = local_model
            self.first_flag = False

    def aggregate(self, agg_info):
        """
        To preform aggregation

        Arguments:
        agg_info (dict): the feedbacks from clients
        :returns: the aggregated results
        :rtype: dict
        """
        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None

        if self.cfg.attack.krum or self.cfg.attack.multi_krum:
            avg_model = self._para_krum_avg(models)
        else:
            avg_model = self._para_weighted_avg(models,
                                                recover_fun=recover_fun)

        return avg_model

    def update(self, model_parameters):
        '''
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        '''
        self.model.load_state_dict(model_parameters, strict=False)

    def allow_benign_model(self, models):
        self.allow_benign_model_init(models)
        if self.general_total_time > 10: return
        norm_list = []
        for i in range(len(models)):
            client_id, local_sample_size, local_model = models[i]
            self.final_model[i] = list(local_model.values())[-1]
            norm_list.append(float(np.linalg.norm(self.final_model[i].float(), ord=2)))
        norm_list = np.array(norm_list)
        norm_list = regular_axis(norm_list)
        cluster_dir_diff_matrix = []
        for i in range(len(models)):
            cluster_dir_diff_list = []
            for j in range(len(models)):
                cluster_dir_diff = direction(self.final_model[i].float(), self.final_model[j].float())
                cluster_dir_diff_list.append(cluster_dir_diff)
            cluster_dir_diff_list = [x * norm_list[i] for x in cluster_dir_diff_list]
            cluster_dir_diff_matrix.append(cluster_dir_diff_list)
        # labels = dbscan_clustering(cluster_dir_diff_matrix)
        kmeans = KMeans(n_clusters=2, random_state=0)
        labels = kmeans.fit_predict(cluster_dir_diff_matrix)

        label_counts = Counter(labels)
        majority_label = label_counts.most_common(1)[0][0]
        minority_label = 1 - majority_label

        weight_flag = np.zeros(len(labels))
        majority_count = label_counts[majority_label]
        for index, label in enumerate(labels):
            if label == majority_label:
                weight_flag[index] = 1
            else:
                weight_flag[index] = 0
        
        self.cluster_benign_model = None
        for i in range(len(models)):
            client_id, local_sample_size, local_model = models[i]
            if weight_flag[i] == 1:
                if self.cluster_benign_model is None:
                    self.cluster_benign_model = list(local_model.values())[-1].clone()
                else:
                    self.cluster_benign_model += list(local_model.values())[-1]
        self.cluster_benign_model /= majority_count

    def weight_aggregation(self, models):
        weight_list = []
        self.allow_benign_model(models)
        self.general_total_time += 1
        for i in range(len(models)):
            client_id, local_sample_size, local_model = models[i]
            pre_value = self.pre_value[client_id]
            self.total_time[client_id] += 1
            final_pre_value = list(pre_value.values())
            final_pre_value = final_pre_value[-1]
            final_local_value = list(local_model.values())
            final_local_value = final_local_value[-1]
            dir_diff = direction(final_pre_value.float(), final_local_value.float())
            penalty_factor = (float(np.linalg.norm(list(local_model.values())[-1].float(), ord=2))*100)
            if dir_diff <= 0:
                dir_diff = dir_diff * penalty_factor
            weight_list.append(dir_diff)
            self.pre_value[client_id] = self.pri_benign[client_id]

        array_weight_list = np.array(weight_list)
        weight_result = regular(array_weight_list)
        return weight_result

    def save_model(self, path, cur_round=-1):
        assert self.model is not None
        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        assert self.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))

    def _para_weighted_avg(self, models, recover_fun=None):
        training_set_size = 0
        for i in range(len(models)):
            client_id, sample_size, _ = models[i]
            training_set_size += sample_size
            # Similarity
            if i == 0:
                target_model = _
        
        weight_result = []
        if self.cfg.attack.robustness:
            weight_result = self.weight_aggregation(models)

        client_id, sample_size, avg_model = models[0]
        for key in avg_model:
            for i in range(len(models)):
                client_id, local_sample_size, local_model = models[i]


                if self.cfg.federate.ignore_weight:
                    weight = 1.0 / len(models)
                elif self.cfg.federate.use_ss:
                    weight = 1.0
                elif self.cfg.attack.robustness:
                    weight = weight_result[i]
                else:
                    weight = local_sample_size / training_set_size

                if not self.cfg.federate.use_ss:
                    if isinstance(local_model[key], torch.Tensor):
                        local_model[key] = local_model[key].float()
                    else:
                        local_model[key] = torch.FloatTensor(local_model[key])

                if i == 0:
                    avg_model[key] = local_model[key] * weight
                else:
                    avg_model[key] += local_model[key] * weight

            if self.cfg.federate.use_ss and recover_fun:
                avg_model[key] = recover_fun(avg_model[key])
                avg_model[key] /= training_set_size
                avg_model[key] = torch.FloatTensor(avg_model[key])

        return avg_model

    def _para_krum_avg(self, models):

        num_workers = len(models)
        num_adv = 1

        num_dps = []
        vectorize_nets = []
        for i in range(len(models)):
            sample_size, local_model = models[i]
            # training_set_size += sample_size
            num_dps.append(sample_size)
            vectorize_nets.append(
                vectorize_net_dict(local_model).detach().cpu().numpy())

        neighbor_distances = []
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i + 1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(float(np.linalg.norm(g_i - g_j)**2))
            neighbor_distances.append(distance)

        # compute scores
        nb_in_score = num_workers - num_adv - 2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])
            # alternative to topk in pytorch and tensorflow
            topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
            scores.append(sum(np.take(dists, topk_ind)))

        if self.cfg.attack.krum:
            i_star = scores.index(min(scores))
            _, aggregated_model = models[
                0]  # slicing which doesn't really matter
            load_model_weight_dict(aggregated_model,
                                   torch.from_numpy(vectorize_nets[i_star]))
            # neo_net_list = [aggregated_model]
            logger.info("Norm of Aggregated Model: {}".format(
                torch.norm(torch.from_numpy(vectorize_nets[i_star])).item()))
            # neo_net_freq = [1.0]
            # return neo_net_list, neo_net_freq
            return aggregated_model

        elif self.cfg.attack.multi_krum:
            topk_ind = np.argpartition(scores,
                                       nb_in_score + 2)[:nb_in_score + 2]

            # we reconstruct the weighted averaging here:
            selected_num_dps = np.array(num_dps)[topk_ind]
            reconstructed_freq = [
                snd / sum(selected_num_dps) for snd in selected_num_dps
            ]

            logger.info("Num data points: {}".format(num_dps))
            logger.info(
                "Num selected data points: {}".format(selected_num_dps))

            aggregated_grad = np.average(np.array(vectorize_nets)[topk_ind, :],
                                         weights=reconstructed_freq,
                                         axis=0).astype(np.float32)
            _, aggregated_model = models[
                0]  # slicing which doesn't really matter
            load_model_weight_dict(aggregated_model,
                                   torch.from_numpy(aggregated_grad))
            # neo_net_list = [aggregated_model]
            logger.info("Norm of Aggregated Model: {}".format(
                torch.norm(torch.from_numpy(aggregated_grad)).item()))
            # neo_net_freq = [1.0]
            # return neo_net_list, neo_net_freq
            return aggregated_model


class NoCommunicationAggregator(Aggregator):
    """"Clients do not communicate. Each client work locally
    """
    def aggregate(self, agg_info):
        # do nothing
        return {}


class OnlineClientsAvgAggregator(ClientsAvgAggregator):
    def __init__(self,
                 model=None,
                 device='cpu',
                 src_device='cpu',
                 config=None):
        super(OnlineClientsAvgAggregator, self).__init__(model, device, config)
        self.src_device = src_device

    def reset(self):
        self.maintained = self.model.state_dict()
        for key in self.maintained:
            self.maintained[key].data = torch.zeros_like(
                self.maintained[key], device=self.src_device)
        self.cnt = 0

    def inc(self, content):
        if isinstance(content, tuple):
            sample_size, model_params = content
            for key in self.maintained:
                self.maintained[key] = (self.cnt * self.maintained[key] +
                                        sample_size * model_params[key]) / (
                                            self.cnt + sample_size)
            self.cnt += sample_size
        else:
            raise TypeError(
                "{} is not a tuple (sample_size, model_para)".format(content))

    def aggregate(self, agg_info):
        return self.maintained


class ServerClientsInterpolateAggregator(ClientsAvgAggregator):
    def __init__(self, model=None, device='cpu', config=None, beta=1.0):
        super(ServerClientsInterpolateAggregator,
              self).__init__(model, device, config)
        self.beta = beta  # the weight for local models used in interpolation

    def aggregate(self, agg_info):
        models = agg_info["client_feedback"]
        global_model = self.model
        elem_each_client = next(iter(models))
        assert len(elem_each_client) == 2, f"Require (sample_size, model_para) \
            tuple for each client, " \
                f"i.e., len=2, but got len={len(elem_each_client)}"
        avg_model_by_clients = self._para_weighted_avg(models)
        global_local_models = [((1 - self.beta), global_model.state_dict()),
                               (self.beta, avg_model_by_clients)]

        avg_model_by_interpolate = self._para_weighted_avg(global_local_models)
        return avg_model_by_interpolate


class FedOptAggregator(ClientsAvgAggregator):
    def __init__(self, config, model, device='cpu'):
        super(FedOptAggregator, self).__init__(model, device, config)
        self.optimizer = get_optimizer(model=self.model,
                                       **config.fedopt.optimizer)

    def aggregate(self, agg_info):
        new_model = super().aggregate(agg_info)

        model = self.model.cpu().state_dict()
        with torch.no_grad():
            grads = {key: model[key] - new_model[key] for key in new_model}

        self.optimizer.zero_grad()
        for key, p in self.model.named_parameters():
            if key in new_model.keys():
                p.grad = grads[key]
        self.optimizer.step()

        return self.model.state_dict()