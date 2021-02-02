import torch
import numpy as np


class PageRankTorchSparseOptimal(torch.nn.Module):
    """
    Class implementing PageRank in PyTorch for sparse graphs.
    """
    def __init__(self, source_indices, target_indices, num_iter,
                 d, num_nodes):
        """
        Args:
            source_indices: The source indices for the graph edges
            target_indices: The target indices for the graph edges
            num_iterations: The number of iterations the algorithm is going to
                            run
            d: The parameter d of the PageRank algorithm
            num_nodes: The number of nodes of the graph
        """
        super(PageRankTorchSparseOptimal, self).__init__(
            source_indices, target_indices, num_iter,
            d, num_nodes)

        self.source_indices = source_indices
        self.target_indices = target_indices
        self.num_iter = num_iter
        self.d = d
        self.num_nodes = num_nodes
        self.node_influence = torch.rand(num_nodes)

    def forward(self):
        for i in range(self.num_iterations):
            self.node_influence = self.node_influence / self.normalize_constant
            self.node_influence = self.node_influence.scatter_add(
                0, self.source_indices, self.node_influence[self.target_indices]) - \
                self.node_influence
            self.node_influence = self.node_influence * self.d + \
                ((1 - self.d) / self.num_nodes)
        return self.node_influence


class PageRankNumpyDense:
    """
    Class implementing PageRank in NumPy for sparse graphs.
    Implementation taken from the wikipedia page:
    https://en.wikipedia.org/wiki/PageRank#Python
    """
    @staticmethod
    def pagerank(self, M, V, num_iter, d):
        """
        Args:
            source_indices: The source indices for the graph edges
            target_indices: The target indices for the graph edges
            num_iterations: The number of iterations the algorithm is going to
                            run
            d: The parameter d of the PageRank algorithm
            num_nodes: The number of nodes of the graph
        """
        N = M.shape[1]
        v = np.random.rand(N, 1)
        v = v / np.linalg.norm(v, 1)
        M_hat = (d * M + (1 - d) / N)
        for i in range(num_iter):
            v = M_hat @ v
        return v


class PageRankTorchDense:
    def __init__(self):
        pass

    def forward(self):
        pass


class PageRankTorchSparseNaive(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
