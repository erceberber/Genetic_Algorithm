import numpy as np
import matplotlib.pyplot as plt

# Optimizasyon sonucu elde edilen patternleri yükle
optimized_patterns = np.load("optimized_patterns.npy")

# Pattern'leri görselleştir
plt.figure(figsize=(15, 3))
for i, pattern in enumerate(optimized_patterns):
    plt.subplot(1, 7, i + 1)
    plt.imshow(pattern, cmap='binary')
    plt.title(f"Pattern {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()
