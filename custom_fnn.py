import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class NNModel(nn.Module):
    def __init__(self, input_dim, class_size, hidden_size=512):
        super(NNModel, self).__init__()
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(input_dim, self.hidden_size)  # 768차원의 cls token을..
        self.l2 = nn.Linear(self.hidden_size, class_size)  # 6개로 classification하기 위해..

    def forward(self, x):
        x = F.relu(self.l1(x), inplace=False)
        return self.l2(x), x.clone().detach()


class CustomNNModel:

    def __init__(self, input_dim, class_size, lr):
        self.first = True
        self.loss_maximum = 0
        self.device = "cuda" # if torch.cuda.is_available() else "cpu"
        self.model = NNModel(input_dim, class_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.input_dim, self.class_size = input_dim, class_size

    def train_by_data(self, batch_data, answers):

        self.model.train()
        self.optimizer.zero_grad()

        batch_data = batch_data.view(-1, self.input_dim).to(self.device)
        answers = answers.to(self.device)

        hypothesis, fnn_hidden_layer = self.model(batch_data)
        loss = self.criterion(hypothesis, answers)
        loss.requires_grad_(True)
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return fnn_hidden_layer

    def test_data(self, batch_data):
        self.model.train()
        self.optimizer.zero_grad()

        batch_data = batch_data.view(-1, self.input_dim).to(self.device)

        hypothesis, fnn_hidden_layer = self.model(batch_data)
        return fnn_hidden_layer


