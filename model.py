import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    '''Convolution-Normalization-Activation Module'''
    def __init__(self, in_channel, out_channel, **kargs):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, bias=False, **kargs)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out


class Yolov1(nn.Module):
    def __init__(self, opt, **kargs):
        super(Yolov1, self).__init__()
        self.structure = get_yolo_structure()
        self.in_channel = 3
        self.darknet = create_darknet(self.in_channel, self.structure)

        self.S = opt.S
        self.B = opt.B
        self.C = opt.num_classes

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5))
        )

    def forward(self, x):
        out = self.darknet(x)
        out = torch.flatten(out, start_dim=1)
        out = self.fc_layer(out)

        return out


def create_darknet(in_channel, structure):
    layers = []

    for x in structure:
        if type(x) == tuple:
            layers += [ConvBNAct(in_channel, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
            in_channel = x[1]

        elif type(x) == str:
            layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

        elif type(x) == list:
            conv1 = x[0]
            conv2 = x[1]
            num_repeats = x[2]

            for _ in range(num_repeats):
                layers += [ConvBNAct(in_channel, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])]
                layers += [ConvBNAct(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])]
                in_channel = conv2[1]

    return nn.Sequential(*layers)


def get_yolo_structure():
    return [
        # tuple = (kernel size, output_channel, stride, padding)
        # M : max-pooling 2x2 stride = 2
        (7, 64, 2, 3),
        "M",
        (3, 192, 1, 1),
        "M",
        (1, 128, 1, 0),
        (3, 256, 1, 1),
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        "M",
        [(1, 256, 1, 0), (3, 512, 1, 1), 4],
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        "M",
        [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
        (3, 1024, 1, 1),
        (3, 1024, 2, 1),
        (3, 1024, 1, 1),
        (3, 1024, 1, 1),
    ]