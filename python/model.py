from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Conv=nn.Conv2d(config["in_channels"], config["out_channels"], config["kernel_size"])
        self.MaxPool=nn.MaxPool2d(config["pool_size"])
        self.layers=nn.Sequential(
            self.Conv,
            nn.ReLU(),
            self.MaxPool
        )

    def forward(self, x):
        return self.layers(x)

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Conv1=ConvBlock(config["conv1"])
        self.Conv2=ConvBlock(config["conv2"])
        self.dropout=nn.Dropout(config["dropout_rate"])
        self.dense_layer=nn.Linear(config["flattened_size"], config["hidden_size"])
        self.output_head=nn.Linear(config["hidden_size"], config["num_classes"])
        self.layers=nn.Sequential(
            self.Conv1,
            self.Conv2,
            nn.Flatten(),
            self.dropout,
            self.dense_layer,
            nn.ReLU(),
            self.output_head
        )

    def forward(self, x):
        return self.layers(x)
