import torch


# define model
def linear_regression(x):
    return torch.matmul(x, w) + b


# loss function
def loss_fn(y_pred, y):
    return torch.mean((y_pred - y) ** 2)



if __name__ == '__main__':
    # data
    x_train = torch.tensor([[1.0], [2.0], [3.0]])
    y_train = torch.tensor([[2.0], [4.0], [6.0]])

    # parameter
    w = torch.tensor([[0.0]], requires_grad=True)
    b = torch.tensor([[0.0]], requires_grad=True)

    # optimizer
    optimizer = torch.optim.SGD([w, b], lr=0.01)

    for epoch in range(100):
        # forward propagation
        y_pred = linear_regression(x_train)
        # compute loss
        loss = loss_fn(y_pred, y_train)
        # back propagation
        loss.backward()
        # update parameters
        optimizer.step()
        # grad to clear
        optimizer.zero_grad()
