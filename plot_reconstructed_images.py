import numpy as np
import matplotlib.pyplot as plt

# 24x24'lük resimleri ve optimize edilen pattern setini yükle
images = [
    np.load(f"generate_binary_images/binary_image_{i+1}_{'Circle' if i==0 else 'Square' if i==1 else 'X' if i==2 else 'Triangle' if i==3 else 'Heart'}.npy")
    for i in range(5)
]
optimized_patterns = np.load("optimized_patterns.npy")

# En iyi patternleri kullanarak yeni resim oluşturma
def reconstruct_image(image, patterns):
    reconstructed = np.zeros_like(image)
    for i in range(0, 24, 3):
        for j in range(0, 24, 3):
            block = image[i:i+3, j:j+3]
            best_pattern = find_best_match(block, patterns)
            reconstructed[i:i+3, j:j+3] = best_pattern
    return reconstructed

# En iyi pattern'i bulma fonksiyonu
def find_best_match(block, patterns):
    best_pattern = None
    min_diff = float("inf")
    for pattern in patterns:
        diff = np.sum(np.abs(block - pattern))
        if diff < min_diff:
            min_diff = diff
            best_pattern = pattern
    return best_pattern

# Her resmi optimize edilmiş patternlerle yeniden oluştur
reconstructed_images = [reconstruct_image(image, optimized_patterns) for image in images]

# Resimleri görselleştir
fig, axs = plt.subplots(2, 5, figsize=(15, 6))

for i, (original, reconstructed) in enumerate(zip(images, reconstructed_images)):
    axs[0, i].imshow(original, cmap='binary')
    axs[0, i].set_title(f"Original {i+1}")
    axs[0, i].axis('off')

    axs[1, i].imshow(reconstructed, cmap='binary')
    axs[1, i].set_title(f"Reconstructed {i+1}")
    axs[1, i].axis('off')

plt.tight_layout()
plt.show()
