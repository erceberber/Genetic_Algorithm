import numpy as np
import random

# 24x24'luk resimlerin yuklenmesi
# images: List[np.ndarray]
def load_images(num_images=5):
    images = []
    for i in range(1, num_images + 1):
        images.append(np.load(f"generate_binary_images/binary_image_{i}_{'Circle' if i==1 else 'Square' if i==2 else 'X' if i==3 else 'Triangle' if i==4 else 'Heart'}.npy"))
    return images

# Birey olustur (7x3x3)
def create_individual():
    # 0 ya da 1 degerlerinden olusan 7 tane 3x3'luk pattern olustur ve dondur
    return np.random.randint(2, size=(7, 3, 3))

# Populasyonu baslat
def initialize_population(pop_size=10):
    # param pop_size: Populasyon boyutu
    # return: pop_size kadar bireyden olusan bir liste dondur
    return [create_individual() for _ in range(pop_size)]

# 3x3'luk bir bloga en cok benzeyen pattern'i bul
def find_best_match(block, patterns):
    # param block: 3x3'luk bir blok
    # param patterns: 7 tane 3x3'luk pattern
    # return: en cok benzeyen pattern'i dondur
    best_pattern = None
    min_diff = float("inf")
    # pattern: 7 tane 3x3'luk patternden bir tanesi
    for pattern in patterns:
        diff = np.sum(np.abs(block - pattern))
        if diff < min_diff:
            min_diff = diff
            best_pattern = pattern
    return best_pattern

# Bireyin fitness'ini hesapla
def calculate_fitness(individual, images):
    # param individual: 7 tane 3x3'luk pattern
    # param images: 24x24'luk resimlerin listesi
    # return: bireyin fitness'ini dondur
    total_loss = 0
    for img in images:
        reconstructed = np.zeros_like(img)
        for i in range(0, 24, 3):
            for j in range(0, 24, 3):
                block = img[i:i+3, j:j+3]
                best_pattern = find_best_match(block, individual)
                reconstructed[i:i+3, j:j+3] = best_pattern
        total_loss += np.sum(np.abs(img - reconstructed))
    return total_loss
    
# Secim (Turnuva yöntemi)
def selection(population, images):
    # param population: populasyon, type: List[np.ndarray], 7x3x3
    # param images: 24x24'luk resimlerin listesi
    # return: en iyi bireyi dondur

    # populasyon dizisinden 2 adet ornek al 
    selected = random.sample(population, 2)
    # fitness degerlerini hesapla
    fitness_values = [calculate_fitness(ind, images) for ind in selected]
    # en iyi bireyi bul ve dondur
    return selected[np.argmin(fitness_values)]

# Çaprazlama
def crossover(parent1, parent2):
    # param parent1: 7 tane 3x3'luk pattern, yani ebeveyn1
    # param parent2: 7 tane 3x3'luk pattern, yani ebeveyn2
    # return: crossover sonrasi olusan yeni birey (cocuk) dondur

    # 7 pattern icinde rastgele bir nokta sec ve ilk kismini parent1'den al, ikinci kismini parent2'den alip birlestir
    point = random.randint(1, 6)  
    child = np.vstack((parent1[:point], parent2[point:]))
    return child

# Mutasyon
def mutate(individual, mutation_rate=0.05):
    # param individual: 7 tane 3x3'luk pattern, yani birey
    # param mutation_rate: mutasyon orani
    # return: mutasyona uğramis birey dondur

    for i in range(7):
        if random.random() < mutation_rate:
            # x ve y 0, 1, 2'den rastgele biri olacak, yani satir ve sutun indexi
            x, y = random.randint(0, 2), random.randint(0, 2) 
            # 0 ise 1, 1 ise 0 yap
            individual[i, x, y] = 1 - individual[i, x, y]  
    return individual

# Genetik algoritma ana döngusu
def genetic_algorithm(generations=50, pop_size=20, mutation_rate=0.05):
    # param generations: nesil sayisi
    # param pop_size: populasyon boyutu
    # param mutation_rate: mutasyon orani
    # return: en iyi bireyi dondur

    # resimleri yukle
    images = load_images()
    # populasyonu baslat
    population = initialize_population(pop_size)
    # fitness degerlerini saklamak icin bir liste olustur
    fitness_history = []
    
    for gen in range(generations):  
        # populasyonu fitness degerine gore sirala
        population = sorted(population, key=lambda ind: calculate_fitness(ind, images))
        
        new_population = []
        for _ in range(pop_size // 2):
            # iki ebeveyn sec
            parent1 = selection(population, images)
            parent2 = selection(population, images)
            # ebeveynleri crossover yaparak iki cocuk olustur
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            # cocuklara mutasyon uygula ve yeni populasyona ekle
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])
        
        # populasyonu yeni populasyon olarak guncelle
        population = new_population
        # yeni populasyondan en iyi bireyin fitness'ini hesapla
        best_fitness = calculate_fitness(population[0], images)
        # fitness degerlerini sonrasi icin sakla
        fitness_history.append(best_fitness)
        print(f"Generation {gen + 1}: Best Fitness = {best_fitness}")
    
    return population[0], fitness_history

# Algoritmayi calistir
best_pattern_set, fitness_hist = genetic_algorithm(generations=35, pop_size=50, mutation_rate=0.05)
np.save("optimized_patterns.npy", best_pattern_set)
np.save("fitness_history.npy", fitness_hist)

