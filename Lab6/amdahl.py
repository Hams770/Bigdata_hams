import numpy as np

serial_fractions = [0.05, 0.10, 0.20, 0.50]
workers = [1, 2, 4, 8, 16, 32, 64]

print(f"{'Workers':>8}", end="")
for fs in serial_fractions:
    print(f"  fs={fs:.0%} ", end="")
print()
print("-" * (8 + 9 * len(serial_fractions)))

for p in workers:
    print(f"{p:>8}", end="")
    for fs in serial_fractions:
        speedup = 1 / (fs + (1 - fs) / p)
        print(f"  {speedup:>6.2f}", end="")
    print()

print()
print("=" * 60)
print("Amdahl's Law Analysis")
print("=" * 60)

# Task 1.2.2: Maximum theoretical speedup for fs = 0.15
fs = 0.15
max_speedup = 1 / fs
print(f"\nTask 1.2.2 — fs = 0.15:")
print(f"  lim(p→∞) S(p) = 1 / fs = 1 / {fs} = {max_speedup:.4f}x")
print(f"  No matter how many workers you add, speedup cannot exceed {max_speedup:.2f}x")

# Task 1.2.3: Workers needed for 80% of theoretical max, fs = 0.10
fs = 0.10
theoretical_max = 1 / fs  # = 10
target = 0.80 * theoretical_max  # = 8.0
# Solve: 1 / (fs + (1-fs)/p) = target  =>  p = (1-fs) / (1/target - fs)
p_needed = (1 - fs) / ((1 / target) - fs)
print(f"\nTask 1.2.3 — fs = 0.10, 80% of theoretical max:")
print(f"  Theoretical max = 1 / {fs} = {theoretical_max:.1f}x")
print(f"  80% target      = {target:.1f}x")
print(f"  Solving for p:  p = (1 - fs) / (1/target - fs)")
print(f"                  p = {1-fs} / ({1/target:.4f} - {fs}) = {p_needed:.2f}")
print(f"  => You need approximately {int(np.ceil(p_needed))} workers.")

# Verify
p_check = int(np.ceil(p_needed))
s_check = 1 / (fs + (1 - fs) / p_check)
print(f"  Verification: S({p_check}) = {s_check:.4f}x  (target was {target:.1f}x)")

print(f"""
Task 1.2.4 — Diminishing returns explanation:
  Amdahl's Law shows that as p grows, the (1-fs)/p term shrinks rapidly, but
  the fixed serial fraction fs remains constant. Beyond a certain point, each
  additional worker reduces parallel time by a smaller absolute amount while
  Spark's coordination costs (shuffle planning, task scheduling, barrier sync)
  grow linearly with the number of workers. The shuffle phase requires every
  worker to exchange partition metadata with the driver, generating O(p) network
  messages. As a result, the real-world overhead eventually outpaces the
  diminishing parallel gains, making more workers counterproductive.
""")