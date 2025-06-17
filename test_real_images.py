import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt


# ==== Mismo modelo que usamos en train_autoencoder.py ====
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_image(path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)  # [1, 3, 32, 32]


def save_image(tensor, path):
    tensor = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    plt.imsave(path, tensor)


def evaluate_images():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar modelo
    model = DenoisingAutoencoder().to(device)
    model.load_state_dict(torch.load("models/autoencoder.pth", map_location=device))
    model.eval()

    noisy_dir = "data/eval/noisy"
    clean_dir = "data/eval/clean"
    output_dir = "outputs/denoised"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, 7):
        noisy_path = os.path.join(noisy_dir, f"{i}.jpg")
        clean_path = os.path.join(clean_dir, f"{i}.jpg")

        # Cargar imágenes
        noisy_img = load_image(noisy_path).to(device)
        clean_img = load_image(clean_path).to(device)

        # Denoising
        with torch.no_grad():
            denoised_img = model(noisy_img)

        # Guardar resultado
        save_image(noisy_img, os.path.join(output_dir, f"{i}_noisy.png"))
        save_image(denoised_img, os.path.join(output_dir, f"{i}_denoised.png"))
        save_image(clean_img, os.path.join(output_dir, f"{i}_clean.png"))

        print(f"Imagen {i} procesada.")


if __name__ == "__main__":
    evaluate_images()
