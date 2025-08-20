This GitHub contains the code for our JMIV [paper](https://arxiv.org/abs/2501.03017) which studies expressivity of Input Convex Neural Networks.

To get started, clone the repository and install ```pnpflow``` via pip

```
cd beyondicnn
pip install -e .
```

You can check convexity of a ReLU NN using the `check_convexity` function from `beyondicnn.check_convexity`. The `beyondicnn.convex_reg` file provides a convex regularisation that can be softly enforced during training.
The ReLU NN is expected to have the structure of `beyondicnn.models.FeedForwardNet`.

Experiments conducted in the first [paper](https://arxiv.org/abs/2501.03017) and its follow-up are in the `expes` folder.


### Acknowledgements

This repository builds upon the [relu_edge_subdivision repo](https://github.com/arturs-berzins/relu_edge_subdivision) which computes the polyhedral complex associated with a ReLU network.
