import matplotlib.pyplot as plt
import numpy as np

y1_values = [95.82, 81.33, 57.00, 64.76, 86.33, 40.40, 27.30, 60.22, 74.00, 74.67, 91.00, 66.33]
y2_values = [76.69, 76.78, 56.41, 64.86, 84.59, 37.85, 35.21, 59.34, 75.68, 75.12, 89.89, 65.75]

group_labels = ["CIFAR-10 Resnet-18", "CIFAR-10 Conv-9", "tabular-benchmark", "blog-feedback", "titanic", "red-wine", "breast-cancer-wisconsin", "diabetes-readmission", "banking-marketing", "adult_income_dataset", "covertype", "higgs"]

group_descriptions = [
	"CIFAR-10: Resnet-18; lay=16, units<=65536 (ch*pix)\n                lin lay=1, units=512",
	"CIFAR-10: Conv-9; lay=6, units=12288 (ch*pix)\n                lin lay=2, units=1024",
	"tabular-benchmark: MLP-5; lay=3, units=64",
	"blog-feedback: MLP-5; lay=3, units=144",
	"titanic: MLP-5; lay=3, units=128",
	"red-wine: MLP-5; lay=3, units=128",
	"breast-cancer-wisconsin: MLP-5; lay=3, units=32",
	"diabetes-readmission: MLP-5; lay=3, units=304",
	"banking-marketing: MLP-6; lay=4, units=128",
	"adult_income_dataset: MLP-5; lay=3, units=256",
	"covertype: MLP-7; lay=5, units=512",
	"higgs: MLP-5; lay=3, units=256",
]

# Convert to arrays
x = np.arange(len(y1_values))
y1 = np.array(y1_values)
y2 = np.array(y2_values)
width = 0.4

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars
b1 = ax.bar(x, y1, width=width, color='blue', label='Full Backprop training')
b2 = ax.bar(x + width, y2, width=width, color='magenta', label='Greedy Layer-Wise Breakaway training')

# Annotate bars
ax.bar_label(b1, fmt='%.1f', padding=3, fontsize=6)
ax.bar_label(b2, fmt='%.1f', padding=3, fontsize=6)

# X-axis group labels
ax.set_xticks(x + width/2)
ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=10)

# Minor ticks
#ax.minorticks_on()
#ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.set_yticks(np.arange(0, 100+0.1, 10.0))

# Labels and title
ax.set_xlabel("dataset")
ax.set_ylabel("test accuracy (Top-1)")
ax.set_title("Greedy Layer-Wise Breakaway vs Full Backprop Training")

# Legend of series, moved to the right above the set descriptions
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 0.95), borderaxespad=0.)

# Adjust layout to make room for both legends and description key on the right
fig.subplots_adjust(right=0.8)

# Add descriptive key box on the right
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
key_text = "\n".join(group_descriptions)
ax.text(1.02, 0.5, key_text, transform=ax.transAxes, fontsize=10, verticalalignment='center', bbox=props)

plt.tight_layout()
plt.show()
