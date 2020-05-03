import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


FILE = 'Collated Collated Sim 3.csv'
FILE2 = 'results/Collated Sim 3.csv'

if __name__ == '__main__':
    try:
        sheet = pd.read_csv(FILE)
    except:
        sheet = pd.read_csv(FILE2)
        
    means = sheet[['Episode','Mean Success', 'Mean Collision', 'Rolling Mean Success', 'Rolling Mean Collision']]
    # logs = sheet[['Log Suc Mean', 'Log Col Mean', 'Log Suc Roll Mean', 'Log Col Roll Mean']]

    plt.plot(means['Episode'], means['Mean Success'], color = 'green', label='Mean Success')
    plt.plot(means['Episode'], means['Mean Collision'], color = 'red', label='Mean Collision')

    plt.plot(means['Episode'], means['Rolling Mean Success'], color = 'green', linestyle=':', label='Rolling Mean Success')
    plt.plot(means['Episode'], means['Rolling Mean Collision'], color='red', linestyle=':', label='Rolling Mean Collision')
    plt.axhline(15, label='Optimum Success')
    plt.axhline(0,label='Optimum Collision', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Number of Aircraft')
    plt.title('Mean Number of Aircraft (Sim 2)')
    plt.legend()

    plt.show()

    # plt.plot(means['Episode'], logs['Log Suc Mean'], color = 'green', label='Log Mean Success')
    # plt.plot(means['Episode'], logs['Log Col Mean'], color = 'red', label='Log Mean Collision')

    # plt.plot(means['Episode'], logs['Log Suc Roll Mean'], color = 'green', linestyle=':', label='Log Rolling Mean Success')
    # plt.plot(means['Episode'], logs['Log Col Roll Mean'], color = 'red', linestyle=':', label='Log Rolling Mean Collision')
    # plt.xlabel('Episode')
    # plt.title('Log Mean Number of Aircraft (Sim 1)')

    # plt.show()