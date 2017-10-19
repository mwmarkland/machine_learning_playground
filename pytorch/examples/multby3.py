#
# Code based on example here:
# https://nicodjimenez.github.io/2017/10/08/tensorflow.html?imm_mid=0f734b&cmp=em-data-na-na-newsltr_ai_20171016
#
import numpy as np
import torch
from torch.autograd import Variable


model= torch.nn.Linear(1,1)
loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for t in range(10000):
    x = Variable(torch.from_numpy(np.random.random((1,1)).astype(np.float32)))
    y = x * 3
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    optimizer.zero_grad()
    # Backpropagation
    loss.backward()
    optimizer.step()
    print(loss.data[0])
# end of loop
