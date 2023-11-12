import torch.onnx
import torch
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.DEFAULT)
dummy_input = torch.randn(128, 3, 224, 224, device="cpu")

# Freeze the parameters of the model
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, 120))

# Set the model to run on the device
model = model.to(device)
model.load_state_dict(torch.load("/Users/tonywu/lechacks/extra_files/model.pt", map_location=torch.device("cpu")))
model.eval()

input_names = ["actual_input_1"] + ["learned_%d" %i for i in range(50)]
output_names = ["output1"]

torch.onnx.export(model, dummy_input, "resnnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
