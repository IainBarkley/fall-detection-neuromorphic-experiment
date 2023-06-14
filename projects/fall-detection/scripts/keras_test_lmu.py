
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys,os
import nengo
from tensorflow.keras.layers import Dense
import keras
import keras_lmu
import tensorflow as tf

sys.path.insert(0,'../networks')
from ldn import LDN

sys.path.insert(0,'../')
from utilities import generate_train_test_split

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

data_dir = '../fall_detection_data/merged_processed/'
subject_file = 'SA06-merged.csv'

# LDN parameters
size_in = 9         # 9 features of the fall detection data
theta = 1.
q = 42

# simulation parameters
dt = 0.001

# load data and generate the train/test split
ddf = pd.read_csv(os.path.join(data_dir,subject_file),index_col=0).drop(['TimeStamp(s)','FrameCounter'],axis=1)
#print(ddf)

train_test_split = 0.8
chunk_size = 3600



chunk_indices = np.arange(0,ddf.shape[0],chunk_size)
train_chunk_indices = np.random.choice(chunk_indices,size = int(len(chunk_indices)*train_test_split))
test_chunk_indices = list(set(chunk_indices)-set(train_chunk_indices))

train_df = pd.DataFrame(columns=ddf.columns)
for idx in train_chunk_indices:
    train_df = pd.concat([train_df,ddf.iloc[idx:idx+chunk_size,:]],axis=0)

test_df = pd.DataFrame(columns=ddf.columns)
for idx in test_chunk_indices:
    test_df = pd.concat([test_df,ddf.iloc[idx:idx+chunk_size,:]],axis=0)

train_xs = train_df[['AccX','AccY','AccZ','GyrX','GyrY','GyrZ','EulerX','EulerY','EulerZ']].to_numpy().astype(float)
lmu_train_xs = LDN(theta=theta, q=q, size_in=size_in).apply(train_xs)
train_ys = train_df[['Fall/No Fall']].to_numpy().astype(int)

test_xs = test_df[['AccX','AccY','AccZ','GyrX','GyrY','GyrZ','EulerX','EulerY','EulerZ']].to_numpy().astype(float)
test_ys = test_df[['Fall/No Fall']].to_numpy().astype(int)



#root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
#
#x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
#x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

num_classes = len(np.unique(train_ys))
train_xs = train_xs.reshape((train_xs.shape[0], train_xs.shape[1], 1))
test_xs = test_xs.reshape((test_xs.shape[0], test_xs.shape[1], 1))

def make_model(input_shape):
    # Defining LMU layer
    lmu_layer = keras_lmu.LMU(
        memory_d=1,
        order=42,
        theta=1,
        hidden_cell=tf.keras.layers.SimpleRNNCell(units=10),
        )

    input_layer = keras.layers.Input(input_shape)
    lmus = lmu_layer(input_layer)
    output_layer = Dense(20)(lmus)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


#print(x_train.shape[1:])
#print(train_xs.shape[1:])
model = make_model(input_shape=test_xs.shape[1:])
keras.utils.plot_model(model, show_shapes=True)

epochs = 20
batch_size = 32


callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = model.fit(
    train_xs,
    train_ys,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)

model = keras.models.load_model("best_model.h5")

test_loss, test_acc = model.evaluate(test_xs, test_ys)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()
