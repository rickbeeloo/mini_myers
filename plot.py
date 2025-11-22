import matplotlib.pyplot as plt

# Target sizes
target = [32, 64, 100, 1000, 10000, 100000]

# µs/query values (from @file_context_0)
mini_k1 = [0.0150, 0.0291, 0.0455, 0.4386, 4.4555, 44.6260]
mini_k4 = [0.0156, 0.0308, 0.0466, 0.4438, 4.4917, 44.5050]

sassy_k1 = [0.7694, 0.8093, 0.8397, 1.3907, 7.8483, 70.6989]
sassy_k4 = [0.8517, 0.9016, 1.0213, 1.9342, 11.9567, 107.4130]

plt.figure(figsize=(10, 6))

# Mini (blue family)
plt.plot(target, mini_k1, marker='o', label="mini k=1", linestyle='-', alpha=0.85, color='tab:blue')
plt.plot(target, mini_k4, marker='o', label="mini k=4", linestyle='--', alpha=0.85, color='tab:blue')

# Sassy (red family)
plt.plot(target, sassy_k1, marker='o', label="sassy k=1", linestyle='-', alpha=0.85, color='tab:red')
plt.plot(target, sassy_k4, marker='o', label="sassy k=4", linestyle='--', alpha=0.85, color='tab:red')

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Target length")
plt.ylabel("µs/query")
plt.title("mini_scan vs sassy")
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("test_data/bench.png")
