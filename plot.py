import matplotlib.pyplot as plt

# Target sizes
target = [32, 64, 100, 1000, 10000, 100000, 1000000, 10000000]

# µs/query values
mini_k1 = [0.0156, 0.0295, 0.0452, 0.4420, 4.4100, 43.8270, 441.0338, 4487.4168]
mini_k4 = [0.0156, 0.0312, 0.0454, 0.4569, 4.4112, 43.9898, 440.8002, 4646.1530]

sassy_k1 = [0.7795, 0.8118, 0.8382, 1.3971, 7.6681, 68.6951, 690.4639, 7992.0650]
sassy_k4 = [0.8712, 0.9175, 1.0285, 1.8889, 11.6288, 106.5944, 1071.7429, 11777.7847]

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
plt.show()
