import numpy as np
import sys
import time

sizes = [10_000, 100_000, 500_000, 1_000_000, 5_000_000]
n_features = 50

print(f"{'Rows':>12} {'Features':>10} {'Matrix MB':>12} {'Gen Time (s)':>14} {'sys.getsizeof':>16} {'nbytes':>12}")
print("-" * 80)

for n in sizes:
    t0 = time.time()
    X = np.random.randn(n, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
    elapsed = time.time() - t0

    mb = X.nbytes / (1024 ** 2)
    sizeof = sys.getsizeof(X)
    print(f"{n:>12,} {n_features:>10} {mb:>12.2f} {elapsed:>14.4f} {sizeof:>16} {X.nbytes:>12}")
    del X, y  # free before next iteration

print()
print("=" * 80)
print("Task 1.1 Notes:")
print("  sys.getsizeof(X) returns only the size of the NumPy array OBJECT (header/metadata)")
print("  X.nbytes returns the actual size of the raw data buffer in bytes.")
print("  Discrepancy: sys.getsizeof gives ~112 bytes (object overhead) regardless of data size.")
print("  X.nbytes = n_rows * n_features * 4 bytes (float32)")
print()

# Predict float64 size for 5M rows
n = 5_000_000
predicted_mb_f64 = (n * n_features * 8) / (1024 ** 2)
print(f"Predicted float64 size for 5M rows: {predicted_mb_f64:.2f} MB (2x float32)")

# Verify
X64 = np.random.randn(n, n_features).astype(np.float64)
actual_mb_f64 = X64.nbytes / (1024 ** 2)
print(f"Actual float64 size for 5M rows:    {actual_mb_f64:.2f} MB")
del X64