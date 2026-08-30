import time
import torch
import torch.nn as nn

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("PS26052 GPU TRAINING BENCHMARK")
print("=" * 60)

print("PyTorch :", torch.__version__)
print("CUDA    :", torch.version.cuda)
print("GPU     :", torch.cuda.get_device_name(0))
print()

# ------------------------------------------------------------
# Same model as TinyEnhancer
# ------------------------------------------------------------

class TinyEnhancer(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


model = TinyEnhancer().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ------------------------------------------------------------
# Same approximate shape produced by 4-second STFT
#
# Frequency bins = 257
# Time frames    = ~501
# Batch size     = 4
# ------------------------------------------------------------

x = torch.randn(
    4, 1, 257, 501,
    device=device
)

target = torch.rand_like(x)

# ------------------------------------------------------------
# Warm-up
# ------------------------------------------------------------

print("Warming up GPU...")

for _ in range(20):

    optimizer.zero_grad()

    output = model(x)

    loss = criterion(output, target)

    loss.backward()

    optimizer.step()

torch.cuda.synchronize()

# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------

iterations = 100

start = time.perf_counter()

for _ in range(iterations):

    optimizer.zero_grad()

    output = model(x)

    loss = criterion(output, target)

    loss.backward()

    optimizer.step()

torch.cuda.synchronize()

elapsed = time.perf_counter() - start

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

step_time = elapsed / iterations
steps_per_second = 1.0 / step_time

memory_mb = (
    torch.cuda.max_memory_allocated()
    / 1024**2
)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Total time       : {elapsed:.3f} sec")
print(f"Training step    : {step_time * 1000:.2f} ms")
print(f"Steps / second   : {steps_per_second:.2f}")
print(f"Peak GPU memory  : {memory_mb:.1f} MB")

print()
print("CUDA benchmark: PASS")
print("=" * 60)
