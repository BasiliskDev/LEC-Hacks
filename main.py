from PIL import Image
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from openai import OpenAI
import os
import time

device = torch.device("cpu")
dir = os.getcwd()
img = Image.open(dir + "/input_file/gsd.jpeg")
model = torchvision.models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.DEFAULT)
# Freeze the parameters of the model
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, 120))
model.to(device)
model.load_state_dict(torch.load(dir + "/extra_files/model.pt", map_location="cpu"))


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
    root= dir + '/extra_files/Images',
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





client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


knowledge = client.files.create(
    file=open("knowledge.pdf", "rb"),
    purpose='assistants'
)

assistant = client.beta.assistants.create(
    name="Dog Trainer",
    instructions="You are helping people take care of all different kinds of dogs." +  
                    "Write how to take care of a dog given the dog's breed.",
    model="gpt-4-1106-preview",
    tools =[{"type": "retrieval"}],
    file_ids=[knowledge.id]
)

thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Can you please tell me how to take care of my new " + 
    prediction.split("-")[1] + "? I am a new owner and am not sure how to take care of dogs."
)

run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

while True:
    time.sleep(1)
    run_status = client.beta.threads.runs.retrieve(
        thread_id = thread.id,
        run_id=run.id
    )

    if run_status.status == "completed":
        os.system("clear")
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        print(messages.data[0].content[0].text.value)
        break
