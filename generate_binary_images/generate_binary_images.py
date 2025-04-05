import numpy as np
import matplotlib.pyplot as plt

# 24x24 boyutunda 5 farkli binary resim olusturalim
size = 24
images = []

# Resim 1: Yuvarlak/Daire
circle = np.zeros((size, size), dtype=np.uint8)
center = size // 2
radius = size // 3
for i in range(size):
    for j in range(size):
        # Daire formulu (x-center)^2 + (y-center)^2 < radius^2
        if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
            circle[i, j] = 1
images.append(("Circle", circle))

# Resim 2: Kare
square = np.zeros((size, size), dtype=np.uint8)
start = size // 3
end = size - start
# Kareyi olustur
square[start:end, start:end] = 1
images.append(("Square", square))

# Resim 3: X sekli
x_shape = np.zeros((size, size), dtype=np.uint8)
for i in range(size):
    for j in range(size):
        # X sekli formulu (|i-j| < 3) or (|i+j-size+1| < 3)
        if abs(i - j) < 3 or abs(i + j - size + 1) < 3:
            x_shape[i, j] = 1
images.append(("X", x_shape))

# Resim 4: Ucgen
triangle = np.zeros((size, size), dtype=np.uint8)
for i in range(size):
    width = int((i / (size-1)) * size)
    start_col = (size - width) // 2
    # Ucgeni olustur
    if width > 0:
        triangle[i, start_col:start_col+width] = 1
images.append(("Triangle", triangle))

# Resim 5: Kalp
heart = np.zeros((size, size), dtype=np.uint8)
center_x = size // 2
center_y = size // 2
for i in range(size):
    for j in range(size):
        # Kalp formulu (x^2 + (y-|x|^0.5)^2 < radius^2)
        x = (j - center_x) / (size / 3)
        y = (i - center_y) / (size / 3)
        if (x*x + (y + 0.5 * abs(x) ** 0.5) ** 2) < 1:
            heart[i, j] = 1
images.append(("Heart", heart))

# Resimleri görsellestirelim ve kaydedelim
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i, (name, img) in enumerate(images):
    axs[i].imshow(img, cmap='binary')
    axs[i].set_title(name)
    axs[i].axis('off')
    
    # Resmi numpy array olarak kaydet
    np.save(f"binary_image_{i+1}_{name}.npy", img)
    
    # Binary değerleri bir text dosyasina yazdiralim
    with open(f"binary_image_{i+1}_{name}.txt", "w") as f:
        for row in img:
            f.write("".join(str(int(pixel)) for pixel in row) + "\n")

plt.tight_layout()
plt.savefig("binary_images.png")
plt.show()

print("Tum resimler basariyla olusturuldu ve kaydedildi!")