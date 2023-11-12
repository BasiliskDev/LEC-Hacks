from PIL import Image
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from openai import OpenAI
import os

device = torch.device("cpu")

img = Image.open("/Users/tonywu/lechacks/input_file/gsd.jpeg")
model = torchvision.models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.DEFAULT)
# Freeze the parameters of the model
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, 120))
model.to(device)
model.load_state_dict(torch.load("/Users/tonywu/lechacks/extra_files/model.pt", map_location="cpu"))


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(
    root='/Users/tonywu/lechacks/extra_files/Images',
    transform=transform
)

img = transform(img)

if torch.cuda.is_available():
    model = model.cuda()
    img = img.cuda()  # Make sure the input data is also on the GPU
else:
    model = model.cpu()
    img = img.cpu()

img = img.unsqueeze(0)

model.eval()
outputs = model(img)
outputs = torch.nn.functional.softmax(outputs, dim=1)
_ , predicted = torch.max(outputs, 1)

prediction = train_dataset.classes[predicted.item()]

print(f'Predicted: {prediction}')



client = OpenAI(api_key="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")

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "Please provide basic info on how to train and take care of a " + prediction.split("-")[1]},
  ]
)

print(completion.choices[0].message.content)