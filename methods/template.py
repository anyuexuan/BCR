import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
import os
from collections import deque
from utils.metrics import evaluation


class MLLTemplate(nn.Module):
    def __init__(self, model_func, n_way=16, n_shot=5, n_query=8,
                 gradient_clip_value=5.0, device='cuda:0', verbose=False):
        super(MLLTemplate, self).__init__()
        self.feature_extractor = model_func()
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.feat_dim = self.feature_extractor.final_feat_dim
        self.gradient_clip_value, self.gradient_norm_queue = gradient_clip_value, deque([np.inf], maxlen=5)
        self.device = device
        self.verbose = verbose

    @abstractmethod
    def set_forward_loss(self, x_support, y_support, x_query, y_query):
        # batch -> loss value
        pass

    @abstractmethod
    def set_forward(self, x_support, y_support, x_query):
        # x -> predicted score
        pass

    def train_loop(self, train_loader):
        self.train()
        num_support = self.n_way * self.n_shot
        avg_loss = 0
        for batch in train_loader:
            x = batch['image'].to(self.device)
            y = batch['labels'].float()

            x_support = x[:num_support]
            y_support = y[:num_support]
            x_query = x[num_support:num_support + self.n_query]
            y_query = y[num_support:num_support + self.n_query]
            y_class = y[num_support + self.n_query:]
            sampled_idx = y_class.sum(0).bool()
            y_support = y_support[:, sampled_idx].to(self.device)
            y_query = y_query[:, sampled_idx].to(self.device)
            self.optimizer.zero_grad()
            loss = self.set_forward_loss(x_support, y_support, x_query, y_query)
            loss.backward()
            self.clip_gradient()
            self.optimizer.step()
            avg_loss += loss.item()
        return avg_loss

    def test_loop(self, test_loader):
        self.eval()
        num_support = self.n_way * self.n_shot
        iter_num = len(test_loader)
        results = {}
        results['mAP'] = []
        for batch in test_loader:
            x = batch['image'].to(self.device)
            y = batch['labels'].float()

            x_support = x[:num_support]
            y_support = y[:num_support]
            x_query = x[num_support:num_support + self.n_query]
            y_query = y[num_support:num_support + self.n_query]
            y_class = y[num_support + self.n_query:]

            sampled_idx = y_class.sum(0).bool()

            y_support = y_support[:, sampled_idx].to(self.device)
            y_query = y_query[:, sampled_idx]

            if y_query.sum() == 0:
                continue

            with torch.no_grad():
                y_pred = self.set_forward(x_support, y_support, x_query)
            if type(y_pred) == torch.tensor:
                y_pred = y_pred.detach().cpu().numpy()
            y_test = y_query.numpy()
            result = evaluation(y_test, y_pred)
            results['mAP'].append(result['mAP'])

        results['mAP-std'] = 1.96 * np.std(results['mAP']) / np.sqrt(iter_num) * 100
        results['mAP'] = np.mean(results['mAP']) * 100
        return results

    def save(self, path, epoch=None, save_optimizer=False):
        os.makedirs(path, exist_ok=True)
        if type(epoch) is str:
            save_path = os.path.join(path, '%s.tar' % epoch)
        elif epoch is None:
            save_path = os.path.join(path, 'model.tar')
        else:
            save_path = os.path.join(path, '%d.tar' % epoch)
        while True:
            try:
                if not save_optimizer:
                    torch.save({'model': self.state_dict(), }, save_path)
                else:
                    torch.save({'model': self.state_dict(),
                                'optimizer': self.optimizer.state_dict(), }, save_path)
                return
            except:
                pass

    def load(self, path, epoch=None, load_optimizer=False):
        if type(epoch) is str:
            load_path = os.path.join(path, '%s.tar' % epoch)
        else:
            if epoch is None:
                files = os.listdir(path)
                files = np.array(list(map(lambda x: int(x.replace('.tar', '')), files)))
                epoch = np.max(files)
            load_path = os.path.join(path, '%d.tar' % epoch)
        tmp = torch.load(load_path)
        self.load_state_dict(tmp['model'])
        if load_optimizer:
            self.optimizer.load_state_dict(tmp['optimizer'])

    def clip_gradient(self):
        if self.gradient_clip_value is not None:
            max_norm = max(self.gradient_norm_queue)
            total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm * self.gradient_clip_value)
            self.gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))

    def cosine_similarity(self, x, y):
        '''
        Cosine Similarity of two tensors
        Args:
            x: torch.Tensor, m x d
            y: torch.Tensor, n x d
        Returns:
            result, m x n
        '''
        assert x.size(1) == y.size(1)
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
        return x @ y.transpose(0, 1)

    def mahalanobis_dist(self, x, y):
        # x: m x d
        # y: n x d
        # return: m x n
        assert x.size(1) == y.size(1)
        cov = torch.cov(x)  # [m,m]
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
        y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
        delta = x - y  # [m,n,d]
        return torch.einsum('abc,abc->ab', torch.einsum('abc,ad->abc', delta, torch.inverse(cov)), delta)

    def euclidean_dist(self, x, y):
        # x: m x d
        # y: n x d
        # return: m x n
        assert x.size(1) == y.size(1)
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
        y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
        return torch.pow(x - y, 2).sum(2)
