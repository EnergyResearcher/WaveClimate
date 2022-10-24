import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
import numpy as np
import pandas as pd
import h5py
import pickle
import glob
import random
import multiprocessing

class gen:
    def __call__(self, file):
        with h5py.File(file, 'r') as hf:
            ds = np.array(hf.get('data').get('table'))
            for row in ds:
                features = np.concatenate((row['values_block_0'][:22], row['values_block_0'][23:]))
                labels = np.array([row['values_block_0'][22]])
                yield np.concatenate((features,labels))

def get_length(files):
    ds_len = 0
    for file in files:
        with h5py.File(file, 'r') as f:
            data = f.get('data').get('table')
            ds_len += data.attrs['NROWS']
    return ds_len

def preprocess_data(path):
    """
    Read the data with a generator, shuffle it and split into train (70%) and validation (30%)
    """
    
    paths = glob.glob(path)
    files = random.sample(paths, 10) # choosing 4 files (out of 54) to test the code 
    ds = tf.data.Dataset.from_tensor_slices(files)
    data = ds.interleave(lambda filename: tf.data.Dataset.from_generator(gen(), output_signature=tf.TensorSpec(shape=(40,),dtype=tf.float32),args=(filename,)), cycle_length=multiprocessing.cpu_count(), block_length=10, num_parallel_calls=multiprocessing.cpu_count())
    def map_features(x):
        return (x[:-1],x[-1])
    prepped = data.map(map_features).shuffle(1000).batch(32).prefetch(1)
    ds_len = get_length(files) *0.7
    train = prepped.take(int(ds_len))
    validation = prepped.skip(int(ds_len))
    return train, validation

def compile_model():
    model = Sequential()
    model.add(Input(shape=(39,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1))
    adam = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()
    rmse = tf.keras.metrics.RootMeanSquaredError()
    mape = tf.keras.metrics.MeanAbsolutePercentageError()
        
    model.compile(optimizer=adam,loss=loss_fn,metrics=[loss_fn,rmse,mape])
    return model
    
def main():
    print('[INFO] get data')
    train, validation = preprocess_data('/work/scratch-nopw/vicab/validation/*.hdf5')
    m = compile_model()
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor='mean_squared_error',
            mode='min',
            patience=5),
        #tf.keras.callbacks.ModelCheckpoint(
         #   'outputs/',
          #  monitor='val_loss',
           # save_best_only=True)            
            ]
    history = m.fit(train, batch_size=1024, epochs=20, validation_data=validation)#, #callbacks=callbacks_list)
    with open('outputs/history', 'wb') as f:
         pickle.dump(history, f)

if __name__=='__main__':
    main()