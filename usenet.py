import torch

from models import OmniglotModel_old

if __name__ == '__main__':
    path = r"D:\Users\Administrator\Documents\JupyterNotebook\reptile-pytorch\log\TJ\5_3\checkpoint\check-0.pth"
    reptile_net = OmniglotModel_old(5)

    checkpoint = torch.load(path,map_location=torch.device('cpu'))

    reptile_net.load_state_dict(checkpoint['meta_net'])
    print(reptile_net)

