## Comparison of Python and Matlab on Wine Quality dataset

PyTorch, pandas, numpy, matplotlib and time libraries were used in the implementation of the model in Python whereas in
MATLAB, the neural net was trained using trainrp which is a network training function that updates weight and bias values
according to the resilient backpropagation algorithm. Through experimentation by changing hidden layer sizes and learning rates,
the parameters which led to the best performance of the model were found iteratively. It is intuitive that the higher the number of
hidden neurons and epochs will lead to longer training time and this was true in this model’s experimentation. During the
experimentation, it was apparent that adjusting learning rates too drastically had an adverse effect on the model’s performance.
The final architecture of the model included 7 hidden-layer neurons, sigmoid function as activation function, 250 epochs and a
learning rate of 0.4. The performance of the 2 implementations are compared in the report.
