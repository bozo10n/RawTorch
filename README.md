TODO
- AutoGrad with DAG. Each node is a tensor and edge a operation. Tracks every operation with requires_grad and backprop just performs chain rule (Have to implement this for layers and all tensor operations but setups done with POC)
- Layers Linear, with forward __call__ function (some what started)
- Cost Functions and optimizers
- More Layers like Conv2D, LSTM, RNN, Sequential, GRU, Embedding, Extend it for Graph neural nets, Liquid Neural Nets, HSTU
- custom cuda kernels for parallel processing

FUTURE GOALS
- Reinforcement learning parallel training/hybrid training 
- adaptive exploration trat
- custom CUDA kernels for rl
- transfer learning
- environment wrappers
- visualization tools 
- customaziable rl building blocks
