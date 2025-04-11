For now I'll be focusing purely on linear algebra, matrix multiplication, dot products all that stuff with shapes and what not, + optimizing those iteratively. i feel like getting better at parallel optimization should be the goal here, in other words this is for me to learn. the ml jargon can come later. this is a computational library for now until im not lazy

TODO
- AutoGrad with DAG. Each node is a tensor and edge a operation. Tracks every operation with requires_grad and backprop just performs chain rule (Have to implement this for layers and all tensor operations but setups done with POC)
- Layers Linear, with forward __call__ function (some what started)
- Cost Functions and optimizers
- More Layers like Conv2D, LSTM, RNN, Sequential, GRU, Embedding, Extend it for Graph neural nets, Liquid Neural Nets, HSTU
- custom cuda kernels for parallel processing
- mix of experts in layers
- state space models
- world models
- Add JAX-style function transformations (vmap, grad)
- Support dynamic computational graphs like PyTorch 2.0

FUTURE GOALS
- Reinforcement learning parallel training/hybrid training 
- adaptive exploration trat
- custom CUDA kernels for rl
- transfer learning
- environment wrappers
- visualization tools 
- customaziable rl building blocks
