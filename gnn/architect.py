import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])  # 把x先拉成一行，然后把所有的x摞起来，变成n行


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model

        # 用来更新α的optimizer，优化的是arch_parameters
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay
        )

        """
          我们更新梯度就是theta = theta + v + weight_decay * theta 
            1.theta就是我们要更新的参数
            2.weight_decay*theta为正则化项用来防止过拟合
            3.v的值我们分带momentum和不带momentum：
              普通的梯度下降：v = -dtheta * lr 其中lr是学习率，dx是目标函数对x的一阶导数
              带momentum的梯度下降：v = lr*(-dtheta + v * momentum)
        """

    def _compute_unrolled_model(self, data, train_idx, eta, network_optimizer):
        """
        计算公式6：w' = w − ξ*dwLtrain(w, α)【完全复制外面的Network更新w的过程】
        不直接用外面的optimizer来进行w的更新，而是自己新建一个unrolled_model展开，主要是因为我们这里的更新不能对Network的w进行更新
        :param input: 训练集数据
        :param target: 训练集数据的标签
        :param eta: 学习率
        :param network_optimizer: 外部的optimizer
        :return: 参数更新为w'之后的模型
        """

        loss = self.model._loss(data, train_idx)  # Ltrain
        theta = _concat(self.model.parameters()).data  # 把参数整理成一行代表一个参数的形式,得到我们要更新的参数theta
        try:
            # momentum*v,用的就是Network进行w更新的momentum
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)  # 不加momentum

        # 前面的是loss对参数theta求梯度，self.network_weight_decay*theta就是正则项
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        # 对参数进行更新，等价于optimizer.step()
        # w − ξ*dwLtrain(w, α)

        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        # unrolled_model = self._construct_model_from_theta(theta.sub(moment + dtheta, alpha=eta))
        return unrolled_model

    def step(self, data, train_idx, valid_idx, eta, network_optimizer, unrolled):
        """

        :param input_train: 训练集数据
        :param target_train: 训练集数据的标签
        :param input_valid: 验证集数据
        :param target_valid: 验证集数据的标签
        :param eta: 学习率
        :param network_optimizer: 外部的optimizer
        :param unrolled: 是否使用本文方法
        """
        self.optimizer.zero_grad()  # 清除上一步的残余更新参数值
        if unrolled:  # 用论文的提出的方法
            self._backward_step_unrolled(data, train_idx, valid_idx, eta, network_optimizer)
        else:  # 不用论文提出的bilevel optimization，只是简单的对α求导
            self._backward_step(data, valid_idx)

        # 应用梯度：根据反向传播得到的梯度进行参数的更新， 这些parameters的梯度是由loss.backward()得到的，optimizer存了这些parameters的指针
        self.optimizer.step()
        # 因为这个optimizer是针对alpha的优化器，所以他存的都是alpha的参数

    def _backward_step(self, data, valid_idx):
        loss = self.model._loss(data, valid_idx)
        loss.backward()  # 反向传播，计算梯度

    def _backward_step_unrolled(self, data, train_idx, valid_idx, eta, network_optimizer):
        """
        计算公式六：dαLval(w',α) ，其中w' = w − ξ*dwLtrain(w, α)
        :param input_train: 训练集数据
        :param target_train: 训练集数据的标签
        :param input_valid: 验证集数据
        :param target_valid: 验证集数据的标签
        :param eta: 学习率
        :param network_optimizer: 外部的optimizer
        """

        # w'
        # unrolled_model里的w已经是做了一次更新后的w，也就是得到了w'
        unrolled_model = self._compute_unrolled_model(
            data=data,
            train_idx=train_idx,
            eta=eta,
            network_optimizer=network_optimizer
        )

        # Lval
        # 对做了一次更新后的w的unrolled_model求验证集的损失，Lval，以用来对α进行更新
        unrolled_loss = unrolled_model._loss(data, valid_idx)
        unrolled_loss.backward()  # todo 为什么使用backward()方法，此时参数不是w吗？假设得到了alpha的梯度

        # dαLval(w',α)
        # 从unrolled_model.arch_parameters中取出alpha的梯度dalpha
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]

        # dw'Lval(w',α)
        # 从unrolled_model.parameters()中取出w'
        vector = [v.grad.data for v in unrolled_model.parameters()]

        # 计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)   其中w+=w+dw'Lval(w',α)*epsilon w- = w-dw'Lval(w',α)*epsilon
        implicit_grads = self._hessian_vector_product(vector, data, train_idx)

        # 公式六减公式八 dαLval(w',α)-(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)
            # g.data.sub_(ig.data, alpha=eta)

        # 对α进行更新
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    # 对应optimizer.step()，对新建的模型的参数进行更新
    def _construct_model_from_theta(self, theta):
        """
        传进来的theta就是 w*, 我们需要将w*更新到我们的模型中
        """
        model_new = self.model.new()
        model_dict = self.model.state_dict()  # Returns a dictionary containing a whole state of the module.

        params, offset = {}, 0
        for k, v in self.model.named_parameters():  # k是参数的名字，v是参数
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())  # 将参数k的值更新为theta对应的值
            offset += v_length

        assert offset == len(theta)  # 确保所有的参数都更新，且前后长度一致
        model_dict.update(params)  # 模型中的参数已经更新为做一次反向传播后的值
        model_new.load_state_dict(model_dict)  # 恢复模型中的参数，也就是我新建的mode_new中的参数为model_dict
        return model_new.cuda()

    # 计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)   其中w+=w+dw'Lval(w',α)*epsilon w- = w-dw'Lval(w',α)*epsilon
    def _hessian_vector_product(self, vector, data, train_idx, r=1e-2):  # vector就是dw'Lval(w',α)
        R = r / _concat(vector).norm()  # epsilon

        # dαLtrain(w+,α)
        for p, v in zip(self.model.parameters(), vector):
            # p.data.add_(R, v)  # 将模型中所有的w'更新成w+=w+dw'Lval(w',α)*epsilon
            p.data.add_(v, alpha=R)
        loss = self.model._loss(data, train_idx)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        # dαLtrain(w-,α)
        for p, v in zip(self.model.parameters(), vector):
            # p.data.sub_(2 * R, v)  # 将模型中所有的w'更新成w- = w+ - (w-)*2*epsilon = w+dw'Lval(w',α)*epsilon - 2*epsilon*dw'Lval(w',α)=w-dw'Lval(w',α)*epsilon
            p.data.sub_(v, alpha=2 * R)
        loss = self.model._loss(data, train_idx)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        # 将模型的参数从w-恢复成w
        for p, v in zip(self.model.parameters(), vector):
            # p.data.add_(R, v)  # w=(w-) +dw'Lval(w',α)*epsilon = w-dw'Lval(w',α)*epsilon + dw'Lval(w',α)*epsilon = w
            p.data.add_(v, alpha=R)
        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
