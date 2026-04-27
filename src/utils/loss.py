import torch
import torch.nn.functional as F



def compute_win_rej(rewards, labels):
    """
    rewards: [b, n]
    chose_labels: [b]
    """
    # TODO
    assert rewards.size(1) == 4

    win_rewards = rewards[range(len(rewards)), labels][:, None] # [b, 1]
    i1 = [[i, i, i] for i in range(len(rewards))]
    i2 = [[i for i in range(4) if i!=l] for l in labels]
    rej_rewards = rewards[i1, i2] # [b, 3]

    return win_rewards - rej_rewards, win_rewards, rej_rewards



def dpo_loss(rewards, ref_rewards, chose_labels, beta):
    """
    Direct Preference Optimization: Your Language Model is Secretly a Reward Model

    rewards, ref_rewards: [b, n]
    chose_labels: [b]
    """
    win_rej = compute_win_rej(rewards, chose_labels)[0]
    ref_win_rej = compute_win_rej(ref_rewards, chose_labels)[0]

    loss = -F.logsigmoid((win_rej - ref_win_rej) * beta).mean()

    return loss



def dpo_loss_wo_ref(rewards, chose_labels, beta, margin=0):
    win_rej = compute_win_rej(rewards, chose_labels)[0]-margin
    loss = -F.logsigmoid(win_rej * beta).mean()
    return loss



def rrhf_loss(rewards, chose_labels):
    """
    RRHF: Rank Responses to Align Language Models with Human Feedback without tears

    rewards: [b, n]
    chose_labels: [b]
    """
    win_rej = compute_win_rej(rewards, chose_labels)[0]

    loss = -win_rej
    loss[loss<0] = 0
    loss = loss.mean()

    return loss



def pro_loss(rewards, chose_labels):
    """
    Preference Ranking Optimization for Human Alignment

    rewards: [b, n]
    chose_labels: [b]
    """
    win_rewards, rej_rewards = compute_win_rej(rewards, chose_labels)[1:]

    neg_temperatures = win_rewards - rej_rewards
    pos_temperatures = torch.max(neg_temperatures, dim=1, keepdim=True).values

    win_rewards = win_rewards * pos_temperatures
    rej_rewards = rej_rewards * neg_temperatures
    
    eps = 1e-10
    pro_loss = -win_rewards + (eps+win_rewards.exp()+rej_rewards.exp().sum(dim=-1, keepdim=True)).log()
    pro_loss = pro_loss.mean()

    return pro_loss



class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
            self,
            temperature=0.5,
            contrast_mode='all',
            base_temperature=0.07,
            loss_scaling_factor=1.0
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.loss_scaling_factor = loss_scaling_factor

    def forward(self, features, labels=None, mask=None):
        """
        Compute loss for model. If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bs, n_views, ...].
            labels: ground truth of shape [bs].
            mask: contrastive mask of shape [bs, bs], mask_{i,j}=1 if sample j has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        labels = torch.tensor(labels, dtype=torch.long) # 转换为tensor形式
        features = torch.nn.functional.normalize(features, p=2, dim=-1)  # 归一化
        device = torch.device('cuda')

        if len(features.shape) < 3:
            raise ValueError('features` needs to be [bs, n_views, ...], at least 3 dimensions are required')

        if len(features.shape) > 3:
            import pdb  # 用于调试 Python 程序，帮助开发者逐行检查代码、设置断点、查看变量值等
            pdb.set_trace() # 启动调试器
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('cannot define both labels and mask')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # 创建一个大小为 batch_size x batch_size 的单位矩阵（对角线为 1，其余为 0）
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device) # torch.eq() 是 PyTorch 中的逐元素相等比较函数,返回一个布尔张量
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature) # 计算 锚点特征（anchor feature） 和 对比特征（contrast feature） 之间的相似度，并通过温度参数（temperature） 对相似度进行缩放

        # for numerical stability, 计算 softmax 或对数 softmax 时，如果输入值（anchor_dot_contrast）过大或过小，可能会导致数值溢出（overflow）或下溢（underflow）
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile make
        mask = mask.repeat(anchor_count, anchor_count)

        # mask-out self-contrast cases, torch.scatter : 用于将0填充到目标张量的指定位置
        logits_mask = torch.scatter(
            torch.ones_like(mask),  # 目标张量
            1,  # dim
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),  # 指定位置的索引
            0  # 填充的值
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return self.loss_scaling_factor * loss


# for test
if __name__ == '__main__':
    torch.manual_seed(42)
    batch_size = 4  # 批量大小
    n_views = 2  # 每个样本的视图数
    feature_dim = 8  # 特征维度

    # 随机数据
    features = torch.randn(batch_size, n_views, feature_dim)  # 4,2,8
    print('before normalize : ', features)
    features = torch.nn.functional.normalize(features, p=2, dim=-1)
    print('after normalize : ', features)
    labels = torch.tensor([0,1,0,1])

    device = torch.device('cuda')
    features = features.to(device)
    labels = labels.to(device)

    sup_loss = SupConLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07, loss_scaling_factor=1.0)
    sup_loss.to(device)

    loss = sup_loss(features, labels)
    print("计算得到的损失:", loss.item())