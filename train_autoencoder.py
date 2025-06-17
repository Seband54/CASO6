import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_dataloaders
import os
import matplotlib.pyplot as plt

# Configuraciones generales
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
MODEL_PATH = "models/autoencoder.pth"


# ======= Autoencoder =======
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # [B, 16, 16, 16]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # [B, 32, 8, 8]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # [B, 16, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),   # [B, 3, 32, 32]
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ======= Entrenamiento =======
def train():
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)
    model = DenoisingAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            output = model(noisy)
            loss = criterion(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * noisy.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # Validación
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                output = model(noisy)
                loss = criterion(output, clean)
                total_val_loss += loss.item() * noisy.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Guardar modelo
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModelo guardado en {MODEL_PATH}")

    # Graficar curva de pérdida
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.grid()
    plt.savefig("outputs/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    train()
