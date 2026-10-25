import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Plot line graph
plt.plot(x, y, marker='o')

# Title and Labels
plt.title("Simple Line Graph")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Show graph
plt.grid(True)
plt.show()