import numpy as np
import time


def simulate_broadcast(source_data, n_workers):
    """
    Worker 0 holds source_data and sends it to all other workers.
    Returns: list of length n_workers, each entry is a copy of source_data.
    Messages sent: (n_workers - 1)  [one send per worker from worker 0]
    """
    result = [None] * n_workers
    result[0] = source_data.copy()          # Worker 0 already has it
    for i in range(1, n_workers):
        result[i] = source_data.copy()      # Worker 0 sends to worker i
    return result


def simulate_reduce(worker_data_list, reduce_fn=np.add):
    """
    Each worker holds a local array. Reduce (sum) at worker 0.
    Returns: the aggregated result at worker 0.
    Messages sent: (n_workers - 1)  [each non-root worker sends to worker 0]
    """
    accumulator = worker_data_list[0].copy()
    for i in range(1, len(worker_data_list)):
        accumulator = reduce_fn(accumulator, worker_data_list[i])
    return accumulator


def simulate_allreduce_naive(worker_data_list):
    """
    Naive all-reduce: worker 0 acts as parameter server —
    collect from all, aggregate, broadcast result back.
    Returns: list of length n_workers, each holding the global sum.

    Message count: (n-1) for reduce  +  (n-1) for broadcast  =  2(n-1)
    Data volume  : 2 * (n-1) * d  bytes  (d = vector size in bytes)
    """
    global_sum = simulate_reduce(worker_data_list)
    return simulate_broadcast(global_sum, len(worker_data_list))


def simulate_allreduce_ring(worker_data_list):
    """
    Ring all-reduce (two-phase):
    Phase 1 — Scatter-Reduce : n-1 rounds, each worker sends a chunk
              to its right neighbour and accumulates the incoming chunk.
    Phase 2 — All-Gather     : n-1 rounds, propagate reduced chunks.
    Returns: list of length n_workers, each holding the global sum.

    Message count: 2 * (n-1)  [same count as naive but smaller per-message data]
    Data volume  : 2 * (n-1) * d/n  per worker  =  2*(n-1)/n * d  total per worker
                   → approaches 2d as n grows  (bandwidth-optimal)
    """
    n = len(worker_data_list)
    d = len(worker_data_list[0])
    chunk_size = d // n

    # Work on copies so originals are not mutated
    buffers = [w.copy() for w in worker_data_list]

    # ── Phase 1: Scatter-Reduce ──────────────────────────────────────
    # After n-1 rounds, buffer[i][chunk i] holds the global sum for that chunk
    for step in range(n - 1):
        for i in range(n):
            send_chunk_idx = (i - step) % n          # chunk this worker sends
            recv_from      = (i - 1) % n             # left neighbour
            recv_chunk_idx = (i - step - 1) % n      # chunk arriving from left

            src_start = send_chunk_idx * chunk_size
            src_end   = src_start + chunk_size if send_chunk_idx < n - 1 else d

            dst_start = recv_chunk_idx * chunk_size
            dst_end   = dst_start + chunk_size if recv_chunk_idx < n - 1 else d

            # "Send" the chunk to the right neighbour (right = (i+1) % n)
            right = (i + 1) % n
            # We accumulate into a temporary array to avoid read-after-write issues
            pass  # defer actual accumulation to after all sends in this step

        # Materialise the round: each worker accumulates the chunk it receives
        new_buffers = [b.copy() for b in buffers]
        for i in range(n):
            recv_from      = (i - 1) % n
            recv_chunk_idx = (i - step - 1) % n

            dst_start = recv_chunk_idx * chunk_size
            dst_end   = dst_start + chunk_size if recv_chunk_idx < n - 1 else d

            send_chunk_idx = recv_chunk_idx               # chunk the left neighbour sent
            src_start      = send_chunk_idx * chunk_size
            src_end        = src_start + chunk_size if send_chunk_idx < n - 1 else d

            new_buffers[i][dst_start:dst_end] += buffers[recv_from][src_start:src_end]
        buffers = new_buffers

    # ── Phase 2: All-Gather ──────────────────────────────────────────
    # After n-1 rounds, every worker has the full global sum in all chunks
    for step in range(n - 1):
        new_buffers = [b.copy() for b in buffers]
        for i in range(n):
            recv_from      = (i - 1) % n
            # Which chunk index arrives from the left this round?
            recv_chunk_idx = (i - step) % n

            dst_start = recv_chunk_idx * chunk_size
            dst_end   = dst_start + chunk_size if recv_chunk_idx < n - 1 else d
            src_start = dst_start
            src_end   = dst_end

            new_buffers[i][dst_start:dst_end] = buffers[recv_from][src_start:src_end]
        buffers = new_buffers

    return buffers


# ── Test harness ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    N_WORKERS   = 4
    VECTOR_SIZE = 8   # must be divisible by N_WORKERS for ring allreduce

    worker_data = [np.random.randint(1, 10, size=VECTOR_SIZE).astype(float)
                   for _ in range(N_WORKERS)]

    print("Worker gradients:")
    for i, d in enumerate(worker_data):
        print(f"  Worker {i}: {d}")

    expected_sum = sum(worker_data)
    print(f"\nExpected global sum: {expected_sum}")

    # ── Broadcast test
    bc = simulate_broadcast(worker_data[0], N_WORKERS)
    assert all(np.allclose(bc[i], worker_data[0]) for i in range(N_WORKERS)), \
        "Broadcast failed!"
    print("\nBroadcast: PASSED")

    # ── Reduce test
    red = simulate_reduce(worker_data)
    assert np.allclose(red, expected_sum), "Reduce failed!"
    print("Reduce: PASSED")

    # ── AllReduce tests
    result_naive = simulate_allreduce_naive(worker_data)
    result_ring  = simulate_allreduce_ring(worker_data)

    print(f"\nNaive all-reduce Worker 0: {result_naive[0]}")
    print(f"Ring  all-reduce Worker 0: {result_ring[0]}")

    assert np.allclose(result_naive[0], expected_sum), "Naive failed!"
    assert np.allclose(result_ring[0],  expected_sum), "Ring failed!"
    print("\nAll assertions passed.")

    # ── Message / bandwidth analysis ─────────────────────────────────
    print("\n" + "="*60)
    print("Communication Cost Analysis")
    print("="*60)

    n = N_WORKERS
    d = VECTOR_SIZE
    bytes_per_float = 4   # float32

    # Naive allreduce
    naive_msgs  = 2 * (n - 1)
    naive_bytes = 2 * (n - 1) * d * bytes_per_float

    # Ring allreduce
    ring_msgs   = 2 * (n - 1)
    ring_bytes_per_worker = 2 * (n - 1) / n * d * bytes_per_float

    print(f"\nNaive all-reduce  (n={n}, d={d}):")
    print(f"  Messages  : 2(n-1)        = {naive_msgs}")
    print(f"  Total data: 2(n-1)*d*4    = {naive_bytes} bytes")

    print(f"\nRing all-reduce   (n={n}, d={d}):")
    print(f"  Messages  : 2(n-1)        = {ring_msgs}")
    print(f"  Data/worker: 2(n-1)/n*d*4 = {ring_bytes_per_worker:.0f} bytes")

    # Large-scale example: n=8, d=1_000_000 floats (float32)
    print("\n" + "-"*60)
    print("Large-scale example: n=8 workers, d=1,000,000 floats (float32)")
    n2, d2 = 8, 1_000_000
    naive_mb  = 2 * (n2 - 1) * d2 * 4 / (1024**2)
    ring_mb   = 2 * (n2 - 1) / n2 * d2 * 4 / (1024**2)
    print(f"  Naive total data transmitted  : {naive_mb:.2f} MB")
    print(f"  Ring  data per worker         : {ring_mb:.2f} MB")
    print(f"  Reduction factor              : {naive_mb / ring_mb:.2f}x less data in ring")

    print("""
Ring all-reduce is bandwidth-optimal because each worker sends and receives
exactly 2*(n-1)/n * d bytes in total — approaching 2d as n grows, regardless
of the number of workers. The naive approach requires the parameter-server
(worker 0) to receive (n-1)*d bytes and then send (n-1)*d bytes, so its
bottleneck bandwidth scales with n. In ring all-reduce, every link in the
ring carries the same fixed load (d/n bytes per chunk per round), so no
single node is a bottleneck and total bandwidth utilisation is maximised.
""")