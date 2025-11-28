from torch import nn, Tensor


class KoopmanNet(nn.Module):
    """
    Base class for all Koopman network
    """
    def __init__(self):
        super(KoopmanNet, self).__init__()

    def x_encoder(self, x: Tensor):
        raise NotImplementedError

    def u_encoder(self, x: Tensor, u: Tensor):
        raise NotImplementedError

    def koopman_operation(self, x_emb: Tensor, u_emb: Tensor):
        raise NotImplementedError

    def x_decoder(self, x_emb: Tensor):
        raise NotImplementedError

    def u_decoder(self, u_emb: Tensor):
        raise NotImplementedError

