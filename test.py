import os

import matplotlib.pyplot as plt
import numpy as np

# Tạo thư mục Image nếu chưa có
os.makedirs("Image", exist_ok=True)

# 1. Tạo 3 biểu đồ Metric vs Epoch
epochs = np.arange(1, 21)
metrics = [
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "BLEU-4",
    "METEOR",
    "BERTScore",
    "MoverScore",
]
colors = ["red", "green", "gold", "orange", "blue", "lime", "purple"]


def generate_metric_curve(model_name, filename, trend_type="normal"):
    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        # Giả lập dữ liệu
        base = 20 + i * 5
        noise = np.random.normal(0, 0.5, len(epochs))
        if trend_type == "normal":
            values = base + 20 * (1 - np.exp(-0.2 * epochs)) + noise
        elif trend_type == "jump":  # Cho Intra-Attention
            values = base + 10 * (1 - np.exp(-0.2 * epochs))
            values[3:] += 15 * (1 - np.exp(-0.5 * (epochs[3:] - 3)))  # Jump at epoch 3
            values += noise
        elif trend_type == "fast":  # Cho Transformer
            values = base + 25 * (1 - np.exp(-0.8 * epochs)) + noise

        plt.plot(epochs, values, label=metric, color=colors[i], linewidth=2)

    plt.title(f"Training Metrics over Epochs - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"Image/{filename}")
    plt.close()


generate_metric_curve(
    "Pointer-Generator Network", "Pointer_Metrics_Curve.png", "normal"
)
generate_metric_curve("Neural Intra-Attention", "Intra_Metrics_Curve.png", "jump")
generate_metric_curve("Transformer", "Transformer_Metrics_Curve.png", "fast")

# 2. Tạo biểu đồ cột so sánh (Bar Chart)
models = ["Pointer-Gen", "Intra-Attn", "Transformer"]
# Dữ liệu giả lập cho 7 metric * 3 model
data = np.array(
    [
        [39.5, 40.5, 42.1],  # ROUGE-1
        [17.2, 18.1, 19.5],  # ROUGE-2
        [36.4, 37.8, 38.2],  # ROUGE-L
        [12.1, 12.5, 14.2],  # BLEU
        [18.5, 19.2, 21.0],  # METEOR
        [85.2, 86.1, 88.5],  # BERTScore
        [58.1, 59.5, 62.0],  # MoverScore
    ]
)

x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, data[:, 0], width, label=models[0], color="skyblue")
plt.bar(x, data[:, 1], width, label=models[1], color="lightgreen")
plt.bar(x + width, data[:, 2], width, label=models[2], color="salmon")

plt.ylabel("Scores")
plt.title("Performance Comparison on Test Set")
plt.xticks(x, metrics)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("Image/Bar_Chart_Comparison.png")
plt.close()

# 3. Tạo các hình ảnh Visual Analysis giả lập
# Heatmap 1
plt.figure(figsize=(6, 6))
data = np.random.rand(10, 10)
plt.imshow(data, cmap="viridis", aspect="auto")
plt.title("Cross-Attention Heatmap")
plt.xlabel("Input Tokens")
plt.ylabel("Generated Tokens")
plt.colorbar()
plt.savefig("Image/Attn_Pointer.png")
plt.close()

# Heatmap 2
plt.figure(figsize=(6, 6))
data = np.random.rand(10, 10)
mask = np.triu(np.ones_like(data, dtype=bool))
data = np.ma.array(data, mask=mask)  # Masked heatmap
plt.imshow(data, cmap="plasma", aspect="auto")
plt.title("Intra-Decoder Attention")
plt.savefig("Image/Attn_Intra.png")
plt.close()

# Multi-head
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for ax in axs.flat:
    ax.imshow(np.random.rand(10, 10), cmap="Blues")
    ax.axis("off")
plt.suptitle("Multi-Head Attention Layers")
plt.savefig("Image/Transformer_Heads.png")
plt.close()

# TSNE Scatter
plt.figure(figsize=(8, 8))
x = np.random.randn(100)
y = np.random.randn(100)
plt.scatter(x, y, c=np.random.rand(100), cmap="tab10", alpha=0.7)
plt.title("T-SNE Projection of Embeddings")
plt.savefig("Image/TSNE_Embedding.png")
plt.close()

print("Đã tạo xong toàn bộ ảnh placeholder trong thư mục Image/")
