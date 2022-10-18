#from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
import numpy as np
import pandas as pd
#import sklearn
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
import keras_tuner
import h5py
import glob
import random

def get_length(files):
    ds_len = 0
    for file in files:
        with h5py.File(file, 'r') as f:
            data = f.get('data').get('table')
            ds_len += data.attrs['NROWS']
    return ds_len

class gen:
    def __call__(self, file):
        with h5py.File(file, 'r') as hf:
            ds = np.array(hf.get('data').get('table'))
            for row in ds:
                features = np.concatenate((row['values_block_0'][:22], row['values_block_0'][23:]))
                labels = np.array([row['values_block_0'][22]])
                yield np.concatenate((features,labels))

def preprocess_data(path):
    """
    Read the data with a generator, shuffle it and split into train (70%) and validation (30%)
    """
    
    paths = glob.glob(path)
    files = random.sample(paths, 4) # choosing 4 files (out of 54) to test the code 
    ds = tf.data.Dataset.from_tensor_slices(files)
    data = ds.interleave(lambda filename: tf.data.Dataset.from_generator(gen(), output_signature=tf.TensorSpec(shape=(40,),dtype=tf.float32),args=(filename,)), 2, 10)
    def map_features(x):
        return (x[:-1],x[-1])
    prepped = data.map(map_features).shuffle(1000).batch(32).prefetch(1)
    ds_len = get_length(files) *0.7
    train = prepped.take(int(ds_len))
    validation = prepped.skip(int(ds_len))
    return train, validation
    

class MyHyperModel(keras_tuner.HyperModel):
    """
    Model for tuning based on the examples in docs: https://keras.io/guides/keras_tuner/getting_started/
    """
    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(39,)))
        #model.add(Dense(24, input_shape=(24,)))
        #Tune the number of layers
        for i in range(hp.Int('num_layers', 1, 4)): # min and max values are inclusive
            model.add(Dense(
                # Tube number of units separately
                units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                activation='relu'
            ))
        if hp.Boolean('dropout'):
            model.add(Dropout(rate=0.25))
        model.add(Dense(1))
        learning_rate = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
        adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError(name='mean_squared_error')
        rmse = tf.keras.metrics.RootMeanSquaredError()
        mape = tf.keras.metrics.MeanAbsolutePercentageError()
        
        model.compile(optimizer=adam,loss=loss_fn,metrics=[loss_fn,rmse,mape])
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs
        )
class MyTuner(keras_tuner.tuners.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
      kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', [128, 258, 512, 1024])
      return super(MyTuner, self).run_trial(trial, *args, **kwargs)

def main():
    print('[INFO] get data')
    train, validation = preprocess_data('/work/scratch-nopw/vicab/validation/*.hdf5')
    #add test data
    hp = keras_tuner.HyperParameters()
    hmodel = MyHyperModel(hp)
    #compile_model()
    
    tuner = MyTuner(
        hypermodel=hmodel,
        objective='mean_squared_error',
        max_epochs=10,
        factor=3,
        hyperband_iterations=5,
        distribution_strategy=tf.distribute.MirroredStrategy(), # parallelise the optimisation
        directory='/home/users/vicab/hpo',
        project_name='test_hpo',
        overwrite=True
    )
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor='mean_squared_error',
            mode='min',
            patience=5),
        tf.keras.callbacks.TensorBoard('tb_logs')]
    print('[INFO] start search')
    tuner.search(train, epochs=50,validation_data=validation, callbacks=callbacks_list)

    best_hps = tuner.get_best_hyperparameters(num_trials=3)[0]
    print(best_hps)
if __name__=='__main__':
    main()