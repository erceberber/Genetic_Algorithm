import matplotlib.pyplot as plt
import numpy as np

fitness = np.load("fitness_history.npy")

# fitness dizisini çiz
plt.plot(fitness, label='Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness Value')
plt.title('Fitness Value Over Generations')
plt.legend()
plt.grid()
plt.savefig("fitness_plot.png")
plt.show()
