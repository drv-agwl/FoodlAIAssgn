import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNWithVector(nn.Module):
    def __init__(self, num_classes, input_vector_dim, mode="classification"):
        super(SimpleCNNWithVector, self).__init__()
        self.input_vector_dim = input_vector_dim
        self.mode=mode

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)

        self.fc_image = nn.Linear(64 * 3 * 3, 1024)

        self.fc_vector = nn.Linear(input_vector_dim, 1024)

        if self.mode=="classification":
            self.fc_final = nn.Linear(2048, num_classes)
        else:
            self.fc_final = nn.Linear(2048, 1)

    def forward(self, rgb, ir, vector):
        image = torch.cat([rgb, ir], dim=1)

        image = self.pool(F.relu(self.bn1(self.conv1(image))))
        image = self.pool(F.relu(self.bn2(self.conv2(image))))
        image = self.pool(F.relu(self.bn3(self.conv3(image))))
        image = self.pool(F.relu(self.bn4(self.conv4(image))))

        image = image.reshape(-1, 64 * 3 * 3)

        image_output = F.relu(self.fc_image(image))
        vector_output = F.relu(self.fc_vector(vector))

        combined_output = torch.cat((image_output, vector_output), dim=1)

        output = self.fc_final(combined_output)

        if self.mode != "classification":
            output = F.relu(output)

        return output