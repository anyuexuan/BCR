from torch import nn
from methods.template import MLLTemplate
import torch


class BCR(MLLTemplate):
    def __init__(self, model_func, n_way, n_shot, n_query, eta=0.01, gamma=0.01,
                 hidden_dim=100, device='cuda:0', verbose=False):
        super(BCR, self).__init__(model_func=model_func, n_way=n_way, n_shot=n_shot, n_query=n_query,
                                  device=device, verbose=verbose)
        self.eta = eta
        self.gamma = gamma
        self.encoder_x = nn.Linear(self.feat_dim, hidden_dim)
        self.encoder_y = nn.Linear(self.n_way, hidden_dim)
        self.encoder_z = nn.Linear(hidden_dim * 2, hidden_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.to(self.device)

    def set_forward(self, x_support, y_support, x_query):
        y_support = y_support.float()
        z_support = self.feature_extractor(x_support)  # [N*K, d]
        z_query = self.feature_extractor(x_query)  # [Q, d]
        weight = y_support / torch.sum(y_support, dim=0, keepdim=True)  # [N*K, N]
        proto = torch.transpose(weight, 0, 1) @ z_support  # [N,d]
        sim = torch.relu(self.cosine_similarity(z_support, proto))  # [N*K, N]
        attention = y_support * sim + 1e-7  # [N*K, N]
        attention = attention / torch.sum(attention, dim=0, keepdim=True)  # [N*K, N]
        proto = torch.transpose(attention, 0, 1) @ z_support  # [N,d]
        scores = -self.euclidean_dist(z_query, proto) / 64
        scores = torch.sigmoid(scores) * 2
        return scores

    def set_forward_loss(self, x_support, y_support, x_query, y_query):
        y_support = y_support.float()
        z_support = self.feature_extractor(x_support)  # [N*K, d]
        z_query = self.feature_extractor(x_query)  # [Q, d]
        weight = y_support / torch.sum(y_support, dim=0, keepdim=True)  # [N*K, N]
        proto = torch.transpose(weight, 0, 1) @ z_support  # [N,d]
        sim = torch.relu(self.cosine_similarity(z_support, proto))  # [N*K, N]
        attention = y_support * sim + 1e-7  # [N*K, N]
        attention = attention / torch.sum(attention, dim=0, keepdim=True)  # [N*K, N]
        proto = torch.transpose(attention, 0, 1) @ z_support  # [N,d]
        scores = -self.euclidean_dist(z_query, proto) / 64
        scores = torch.sigmoid(scores) * 2
        loss_cls = nn.BCELoss()(scores, y_query)
        # ------------------------ LE loss ------------------------
        x = torch.cat([z_support, z_query], dim=0)
        y = torch.cat([y_support, y_query], dim=0)
        dx = self.encoder_x(x)  # [N*K+Q, hidden]
        dy = self.encoder_y(y)  # [N*K+Q, hidden]
        dz = self.encoder_z(torch.concat([dx, dy], dim=1))  # [N*K+Q, hidden]
        S = self.cosine_similarity(dz, dz)  # [N*K+Q, N*K+Q]
        yy = S @ y  # [N*K+Q, N]
        loss_cl = nn.BCEWithLogitsLoss()(yy, y)
        weight = y / torch.sum(y, dim=0, keepdim=True)  # [N*K+Q, N]
        proto = torch.transpose(weight, 0, 1) @ x  # [N,d]
        sim = torch.relu(self.cosine_similarity(x, proto))  # [N*K+Q, N]
        attention = y * sim + 1e-7  # [N*K+Q, N]
        attention = attention / torch.sum(attention, dim=0, keepdim=True)  # [N*K, N]
        proto = torch.transpose(attention, 0, 1) @ x  # [N,d]
        sscores = -self.euclidean_dist(x, proto) / 64
        loss_li = nn.CrossEntropyLoss()(sscores, torch.softmax(yy, dim=1))
        return loss_cls + loss_cl * self.eta + loss_li * self.gamma
