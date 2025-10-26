from train import run_training
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Starting MNIST MLP training on {device}")
    model = run_training(
        epochs=10,
        batch_size=128,
        lr=1e-3,
        model_path="mlp_best.pth"
    )
    print("🎉 Training finished successfully.")