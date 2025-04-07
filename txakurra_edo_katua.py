import torch
from torchvision import transforms
from PIL import Image

# Kargatu aurretik entrenatutako eredua
modeloa = torch.load('ruta_al_modelo/model.pth')
modeloa.eval()

# Irudiak behar dituen transformazioak
transformazioa = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Irudia kargatu eta transformazioak aplikatu
irudia = Image.open('ruta_a_la_imagen/imagen.jpg')
irudia = transformazioa(irudia).unsqueeze(0)

# Emaitza kalkulatu
with torch.no_grad():
    irteera = modeloa(irudia)
    _, emaitza = torch.max(irteera, 1)
    animali_mota = 'Txakurra' if emaitza.item() == 1 else 'Katua'

print(f'Irudia {animali_mota} bat da')
