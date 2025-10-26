from train import run_training

if __name__ == "__main__":
    # Windows 下必须加这个保护语句
    print("🚀 Starting MNIST MLP training...")
    model = run_training(
        epochs=10,
        batch_size=128,
        lr=1e-3,
        model_path="mlp_best.pth"
    )
    print("🎉 Training finished successfully.")