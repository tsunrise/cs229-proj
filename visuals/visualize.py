import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw(folder, metric):
    logistic = pd.read_csv("visuals/time_series/" + folder + "/" + folder + "_logistic_0.5.csv")
    nn = pd.read_csv("visuals/time_series/" + folder + "/" + folder + "_nn_0.5.csv")
    distilbert = pd.read_csv("visuals/time_series/" + folder + "/" + folder + "_distilbert.csv")

    plt.figure()
    plt.xlim([0, 200])
    plt.ylim([0, 1.05])
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    # plt.title('Precision-Recall Curve')
    
    plt.plot(logistic["Step"], logistic["Value"], label='Logistic')
    plt.plot(nn["Step"], nn["Value"], label='NN')
    plt.plot(distilbert["Step"], distilbert["Value"], label='Distil-BERT')
    plt.legend(loc="lower right")
    plt.savefig(f'visuals/time_series/{folder}.png')


if __name__ == "__main__":
    draw("train_accept", "Accept Rate")
    draw("train_f1", "F1 Score")
    draw("val_accept", "Accept Rate")
    draw("val_f1", "F1 Score")
