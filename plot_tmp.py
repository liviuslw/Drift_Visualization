import matplotlib.pyplot as plt
import pandas as pd

x = pd.read_csv('Data.csv').values
plt.plot(x)
plt.show()