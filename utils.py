import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def save_and_plot_predictions(predictions, labels, epoch):
    # Stack the list of numpy arrays
    predictions_np = np.hstack(predictions)
    labels_np = np.hstack(labels)

    # Save to CSV
    df = pd.DataFrame(data={"Predictions": predictions_np, "Ground Truth": labels_np})
    df.to_csv(f"predictions_{epoch}.csv", index=False)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(df['Predictions'], label='Predictions')
    plt.plot(df['Ground Truth'], label='Ground Truth')
    plt.title('Predictions vs Ground Truth')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(f'pre_{epoch}.png')
