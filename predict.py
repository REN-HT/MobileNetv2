import torch
from PIL import Image
from torchvision import transforms as T
from model.mobileNetv2 import mobileNetv2


def predict():
    model=mobileNetv2()
    model_state_dict=torch.load('your_model.pth')
    model.load_state_dict(model_state_dict)

    path='your_path'
    img=Image.open(path)
    transform=T.Compose([
                         T.Resize(224),
                         T.CenterCrop(224),
                         T.ToTensor(),
                         T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
                         ])
    img=transform(img).unsqueeze(0)
    output=model(img)
    # output=output.clamp(min=0)
    # output=F.softmax(output, dim=1)
    print(output)

if __name__ == '__main__':
    predict()
