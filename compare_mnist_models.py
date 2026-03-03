"""
Train both the original and improved MNIST CNNs with the same seed, then print
a side-by-side comparison of test accuracy and training time.
"""
import torch

import mnist_cnn_original
import mnist_cnn_improved

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("=" * 60)
    print("MNIST CNN: Original vs Improved")
    print("=" * 60)
    print(f"Device: {DEVICE}\n")

    print("Training ORIGINAL model...")
    original = mnist_cnn_original.run(device=DEVICE, seed=SEED)
    print(f"  Test accuracy: {original['test_acc']:.2%}")
    print(f"  Train time:    {original['train_time_sec']:.1f}s\n")

    print("Training IMPROVED model...")
    improved = mnist_cnn_improved.run(device=DEVICE, seed=SEED)
    print(f"  Test accuracy: {improved['test_acc']:.2%}")
    print(f"  Train time:    {improved['train_time_sec']:.1f}s\n")

    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Original':>12} {'Improved':>12} {'Delta':>12}")
    print("-" * 61)
    acc_delta = improved["test_acc"] - original["test_acc"]
    print(f"{'Test accuracy':<25} {original['test_acc']:>11.2%} {improved['test_acc']:>11.2%} {acc_delta:>+11.2%}")
    time_delta = improved["train_time_sec"] - original["train_time_sec"]
    print(f"{'Train time (s)':<25} {original['train_time_sec']:>11.1f} {improved['train_time_sec']:>11.1f} {time_delta:>+10.1f}s")
    print(f"{'Epochs':<25} {original['epochs']:>12} {improved['epochs']:>12}")
    print("=" * 60)


if __name__ == "__main__":
    main()
