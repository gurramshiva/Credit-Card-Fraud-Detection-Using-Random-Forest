import matplotlib.pyplot as plt
import numpy as np

def plot_graph(total, clean, fraud):
    values = [total, clean, fraud]
    labels = ['Total', 'Normal', 'Fraud']
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, values)
    plt.xticks(y_pos, labels)
    plt.title("Transaction Classification")
    plt.show()
