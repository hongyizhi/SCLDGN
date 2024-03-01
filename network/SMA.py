import torch
import copy


class Algorithm(torch.nn.Module):

    def __init__(self,):
        super(Algorithm, self).__init__()

    def update(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class MovingAvg:
    def __init__(self, network, start):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = start
        self.global_iter = 0
        self.sma_count = 0

    def update_sma(self):
        self.global_iter += 1
        new_dict = {}
        if self.global_iter >= self.sma_start_iter:
            self.sma_count += 1
            for (name, param_q), (_, param_k) in zip(self.network.state_dict().items(),
                                                     self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    new_dict[name] = (
                                (param_k.data.detach().clone() * self.sma_count + param_q.data.detach().clone()) / (
                                    1. + self.sma_count))
        else:
            for (name, param_q), (_, param_k) in zip(self.network.state_dict().items(),
                                                     self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    new_dict[name] = param_q.detach().data.clone()
        self.network_sma.load_state_dict(new_dict)


class ERM_SMA(Algorithm, MovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """

    def __init__(self, net, start=100):
        Algorithm.__init__(self)
        self.network = net
        MovingAvg.__init__(self, self.network, start=start)

    def update(self, x):
        y, feature, proj = self.network(x)
        self.update_sma()
        return y, feature, proj

    def predict(self, x):
        self.network_sma.eval()
        return self.network_sma(x)
