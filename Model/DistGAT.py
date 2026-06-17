import time
import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
import tqdm


class DistGAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, n_heads):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.GATConv(in_feats, n_hidden, num_heads=n_heads))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.GATConv(n_hidden * n_heads, n_hidden, num_heads=n_heads))
            self.layers.append(dglnn.GATConv(n_hidden * n_heads, n_classes, num_heads=1))
        else:
            self.layers.append(dglnn.GATConv(in_feats, n_classes, num_heads=1))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
                h = h.flatten(1)  # Flatten the output for multi-head attention
            else:
                h = h.mean(dim=1)
        return h

    def inference_n_layer(self, g, x, seed, device, batch_size, n_layer):
        """
        Modified inference function to compute the output of a specific layer.
        Args:
            n_layer: Specify the layer to compute (0-indexed).
        """
        t_load = []  # Communication time
        t_infer = []  # Computation time

        # Initialize the output matrix based on the specified layer's output dimension
        output_dim = self.n_hidden * self.n_heads if n_layer != len(self.layers) - 1 else self.n_classes
        y = torch.zeros(g.num_nodes(), output_dim)

        layer = self.layers[n_layer]  # Compute only the specified layer

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)

        dataloader = dgl.dataloading.DataLoader(
            g,
            seed,
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

        t0 = time.time()
        pbar = tqdm.tqdm(dataloader)

        for step, (input_nodes, output_nodes, blocks) in enumerate(pbar):
            block = blocks[0]
            block = block.int().to(device)
            h = x[input_nodes].to(device)

            # Record loading/communication time
            t_load.append(time.time() - t0)

            t0 = time.time()

            # Forward computation
            layer.eval()
            h = layer(block, h)
            if n_layer != len(self.layers) - 1:
                h = self.activation(h)
                h = h.flatten(1)  # Flatten the output for multi-head attention
            else:
                h = h.mean(dim=1)
            y[output_nodes] = h.cpu()

            # Record inference/computation time
            t_infer.append(time.time() - t0)
            t0 = time.time()

        return y, sum(t_load), sum(t_infer)