import matplotlib.pyplot as plt
import ast  

file_path = 'output.txt'  
with open(file_path, 'r') as file:
    data = file.read()
    # Split the data by commas and evaluate each part as a matrix
    matrix_parts = data.split(',\n')
    matrices = [ast.literal_eval(part.strip()) for part in matrix_parts if part.strip()]

print(len(matrices))
# Plot each matrix in a separate figure
for i, matrix in enumerate(matrices):
    plt.figure(i)
    plt.imshow(matrix, cmap='plasma', aspect='auto') #, vmin=0, vmax=1
    plt.colorbar()  # Adds a color bar to the side
    if i % 3 == 0:
        time = "Initial"
    elif i % 3 == 1:
        time = "T/2"
    else:
        time = "T"
    plt.title(f"Distributed Memory Parallel Lax {time}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()