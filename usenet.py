import torch

from models import TmpModel

if __name__ == '__main__':
    path = r"D:\Users\Administrator\Documents\JupyterNotebook\reptile-pytorch\log\IITD\10_3\checkpoint\check-90000.pth"
    reptile_net = TmpModel(imgc=1,imgsz=84)

    checkpoint = torch.load(path,map_location=torch.device('cpu'))

    reptile_net.load_state_dict(checkpoint['meta_net'],strict=False)
    print(reptile_net)

