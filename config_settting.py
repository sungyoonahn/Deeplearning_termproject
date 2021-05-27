import torch

def test_optim(i, model, learning_rate):

    if i==0:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        name = "Adam"
    elif i==1:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        name = "AdamW"
    elif i==2:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        name = "RMSprop"
    elif i==3:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        name = "SGD"

    return optimizer, name