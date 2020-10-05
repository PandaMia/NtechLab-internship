import os
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import datasets

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description='Test prediction')
parser.add_argument('indir', type=str, help='Input dir for images')

args = parser.parse_args()

data_path = args.indir

# для загрузки через DataLoader нужно указывать не путь к изображениям, а путь к папке с изображениями, чтобы присвоить метку класса
split_path = data_path.split('/')[:-1]
path_for_load = ''
for folder in split_path:
    path_for_load = path_for_load + folder + '/'

items = os.listdir(f'./{data_path}') # имена файлов

image_transforms = transforms.Compose([transforms.Resize((64,64)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                           std=(0.5, 0.5, 0.5))
                                      ])

test_set = datasets.ImageFolder(f'./{path_for_load}', transform=image_transforms)

test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool2d(3)
        self.drop2d = nn.Dropout2d(p=0.25)
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=128, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop2d(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop2d(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.drop2d(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

model = Net().to(device)
model.load_state_dict(torch.load('./model_state-epoch-20-val_acc-0.9619.pth'))

model.eval()

predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device=device, non_blocking=True)
        outputs = model(inputs)
        pred = outputs.argmax(1, keepdim=True)
        predictions.append('female' if pred.item() == 0 else 'male')

results = dict(zip(items, predictions))

with open('process_results.json', 'w') as f:
    json.dump(results, f)
