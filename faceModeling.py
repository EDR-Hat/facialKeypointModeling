import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from tensorflow.keras.datasets.mnist import load_data
from google.colab import files
import math
import keras

files.upload() #used google Collab to upload KeypointsDatasetSmall.csv

faces = pd.read_csv("KeypointsDatasetSmall.csv")
faces.dropna(inplace=True)
faces.reset_index(inplace=True)
images = faces["Image"]

def conversion(k):
  h = k.split()
  o = []
  for i in range(len(h)):
    o.append(int(h[i]))
  return np.array(o).reshape(96, 96)

graphfaces = []
for i in range(len(images)):
  graphfaces.append(conversion(images[i]))
graphfaces = np.array(graphfaces)

x_positions = faces.filter(regex="_x$")
y_positions = faces.filter(regex="_y$")

positions = faces.filter(regex="_.$")

positions.columns
positions.shape
positions

model = keras.models.Sequential([
                                 keras.layers.Flatten(input_shape=[96, 96]),
                                 keras.layers.Dense(64, activation="elu"),
                                 keras.layers.Dense(128, activation="relu"),
                                 keras.layers.Dense(128, activation="relu"),
                                 keras.layers.Dense(256, activation="elu"),
                                 keras.layers.Dense(512, activation="relu"),
                                 keras.layers.Dense(30, activation=None)
])

model.compile(loss="mse", optimizer='adam')

ex = positions[:][:496]
imgs = graphfaces[:496]
history = model.fit(imgs, ex, epochs=50)
model.summary()
output = model.predict(graphfaces[496:])
output
df_out = pd.DataFrame(data=output, columns=positions.columns)
out_x = df_out.filter(regex="_x$")
out_y = df_out.filter(regex="_y$")

plt.imshow(graphfaces[496])
plt.plot(out_x.loc[0], out_y.loc[0], 'ro')

#not too bad but still not the most accurate facial keypoint model
