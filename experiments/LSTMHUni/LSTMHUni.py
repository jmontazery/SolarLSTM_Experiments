#!/usr/bin/env python
# coding: utf-8

# Standard libraries
import os
import math

# Importing libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning - Scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, ParameterGrid, learning_curve, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder

# Deep Learning - TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional, Flatten, ConvLSTM2D, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
import json

# In[1]

# Function to create time steps
def create_time_steps(length):
    return list(range(-length, 0))

# Function to create data for multivariate forecasting
def mutlivariate_data(dataset, target, start_idx, end_idx, seq_size, target_size, step, single_step=False):
    data = []
    labels = []
    start_idx = start_idx + seq_size
    if end_idx is None:
        end_idx = len(dataset) - target_size
    for i in range(start_idx, end_idx):
        idxs = range(i - seq_size, i, step)
        data.append(dataset[idxs])
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])
    return np.array(data), np.array(labels)


def prepare_datasets(train, test, validation_split, seq_size, future_target, STEP, batch_size,feature_index):
    train_stop = int(validation_split*len(train))
    
    x_train_multi, y_train_multi = mutlivariate_data(
        train, train[:, feature_index], 0, train_stop, seq_size, future_target, STEP)
    
    x_val_multi, y_val_multi = mutlivariate_data(
        train, train[:, feature_index], train_stop, None, seq_size, future_target, STEP)
    
    x_test_multi, y_test_multi = mutlivariate_data(
        test, test[:, feature_index], 0 , None, seq_size, future_target, STEP)

    print(f'Training Observations: {len(y_train_multi)}')
    print(x_train_multi.shape)
    print(y_train_multi.shape)
    print(f'Validation Observations: {len(y_val_multi)}')
    print(x_val_multi.shape)
    print(y_val_multi.shape)
    print(f'Testing Observations: {len(y_test_multi)}')
    print(x_test_multi.shape)
    print(y_test_multi.shape)
    
    total_train_samples = x_train_multi.shape[0]
    total_val_samples = x_val_multi.shape[0]
    total_test_samples = x_test_multi.shape[0]
    
    train_steps = math.ceil(total_train_samples / batch_size)
    validation_steps = math.ceil(total_val_samples / batch_size)
    test_steps = math.ceil(total_test_samples / batch_size)
    
    print("Train steps:", train_steps)
    print("Validation steps:", validation_steps)
    print("Test steps:", test_steps)
    shape = x_train_multi.shape
    
    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    
    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.cache().batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    
    test_data_multi = tf.data.Dataset.from_tensor_slices((x_test_multi, y_test_multi))
    test_data_multi = test_data_multi.cache().batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    
    print(train_data_multi, val_data_multi, test_data_multi)

    return train_data_multi, val_data_multi, test_data_multi, shape, y_train_multi, y_val_multi, y_test_multi, train_steps, validation_steps, test_steps




def plot_loss(history, title, model_type, index):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, loss, 'b', label='Train Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title(f"{title} - {model_type}")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(plots_save_dir, f"{index}_{model_type}_ loss.jpg")
    plt.savefig(plot_filename)
    plt.show()
    


# Function to plot time series data for multi-step ahead forecast
def multi_step_plot(seq_size, true_future, prediction,model_type, index,i):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(seq_size))
    num_out = len(true_future)
    plt.grid(True)
    plt.plot(num_in, np.array(seq_size[:, 1]), label='seq_size')
    plt.plot(np.arange(num_out) / STEP, np.array(true_future), 'bo', label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out) / STEP, np.array(prediction), 'ro', label='Predicted Future')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plot_filename = os.path.join(plots_save_dir, f"{index}_{model_type}_{i}_Take.jpg")
    plt.savefig(plot_filename)
    plt.show()

def create_multistep_model(best_params, model_type, seq_size, future_target, shape):
    model = tf.keras.models.Sequential()
    units = int(best_params['units'])
    dropout = best_params['dropout']
    dense_units = int(best_params['dense_units'])
    learning_rate = best_params['learning_rate']

    if model_type == 'single_lstm':
        model.add(tf.keras.layers.LSTM(units, return_sequences=False, input_shape=shape[-2:]))
        model.add(tf.keras.layers.Dropout(dropout))
        #model.add(tf.keras.layers.Dense(dense_units, activation='linear'))
    elif model_type == 'stacked_lstm':
        model.add(tf.keras.layers.LSTM(units, return_sequences=True, input_shape=shape[-2:]))
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.LSTM(units))
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(dense_units, activation='linear'))
    elif model_type == 'bidirectional_lstm':
        model.add(Bidirectional(tf.keras.layers.LSTM(units), input_shape=shape[-2:]))
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(dense_units, activation='linear'))
    
    model.add(Dense(future_target, activation='relu'))# Output layer for regression
    #model.add(Dense(future_target, activation='linear'))# Output layer for regression
    optimizer = tf.keras.optimizers.RMSprop(clipvalue=1.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model

def train_and_evaluate_multistep_model(train_data, val_data, test_data, train_steps, validation_steps, test_steps, model, seq_size, future_target, STEP, EPOCHS, batch_size=30, step_evaluate=45, shuffle=False):
    first_layer_type = type(model.layers[0]).__name__
    model_name = 'Model: ' + first_layer_type
    es = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, verbose=1, mode='auto', restore_best_weights=True)
    mc = ModelCheckpoint('best_model.keras', monitor='val_loss', mode='auto', save_best_only=True, verbose=1)

    history = model.fit(train_data, epochs=epochs, batch_size=batch_size, 	steps_per_epoch=train_steps,
                        shuffle=False, verbose=1, callbacks=[es, mc], validation_data=val_data, validation_steps=validation_steps)
    #plot_loss(history, 'Multi-Step Training and validation loss', model_name.replace(' ', '_'), model_name)
    
    return model,history





def plot_error_metrics_over_time(y_train, train_pred, y_val, val_pred, y_test, test_pred, num_timesteps, index, model_type):
    # Initialize arrays to store metrics
    train_mae = []
    train_mse = []
    train_rmse = []
    train_r2 = []
    val_mae = []
    val_mse = []
    val_rmse = []
    val_r2 = []
    test_mae = []
    test_mse = []
    test_rmse = []
    test_r2 = []

    # Calculate metrics for each timestep
    for i in range(num_timesteps):
        train_mae.append(mean_absolute_error(y_train[:, i], train_pred[:, i]))
        train_mse.append(mean_squared_error(y_train[:, i], train_pred[:, i]))
        train_rmse.append(np.sqrt(train_mse[-1]))
        train_r2.append(r2_score(y_train[:, i], train_pred[:, i]))
        
        val_mae.append(mean_absolute_error(y_val[:, i], val_pred[:, i]))
        val_mse.append(mean_squared_error(y_val[:, i], val_pred[:, i]))
        val_rmse.append(np.sqrt(val_mse[-1]))
        val_r2.append(r2_score(y_val[:, i], val_pred[:, i]))

        test_mae.append(mean_absolute_error(y_test[:, i], test_pred[:, i]))
        test_mse.append(mean_squared_error(y_test[:, i], test_pred[:, i]))
        test_rmse.append(np.sqrt(test_mse[-1]))
        test_r2.append(r2_score(y_test[:, i], test_pred[:, i]))

    # Create a DataFrame to store the results
    metrics_df = pd.DataFrame({
        'Time Step': range(num_timesteps),
        'Train MAE': train_mae,
        'Train MSE': train_mse,
        'Train RMSE': train_rmse,
        'Train R2': train_r2,
        'Validation MAE': val_mae,
        'Validation MSE': val_mse,
        'Validation RMSE': val_rmse,
        'Validation R2': val_r2,
        'Test MAE': test_mae,
        'Test MSE': test_mse,
        'Test RMSE': test_rmse,
        'Test R2': test_r2
    })

    # Save the results to a CSV file
    results_filename = os.path.join(plots_save_dir, f"{index}_{model_type}_error_metrics.csv")
    metrics_df.to_csv(results_filename, index=False)

    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(12, 24), sharex=True)

    # MAE plot
    axs[0].plot(train_mae, label='Train MAE', color='blue')
    axs[0].plot(val_mae, label='Validation MAE', color='green')
    axs[0].plot(test_mae, label='Test MAE', color='red')
    axs[0].set_title('Mean Absolute Error over Time Steps')
    axs[0].set_ylabel('MAE')
    axs[0].legend()

    # MSE plot
    axs[1].plot(train_mse, label='Train MSE', color='blue')
    axs[1].plot(val_mse, label='Validation MSE', color='green')
    axs[1].plot(test_mse, label='Test MSE', color='red')
    axs[1].set_title('Mean Squared Error over Time Steps')
    axs[1].set_ylabel('MSE')
    axs[1].legend()

    # RMSE plot
    axs[2].plot(train_rmse, label='Train RMSE', color='blue')
    axs[2].plot(val_rmse, label='Validation RMSE', color='green')
    axs[2].plot(test_rmse, label='Test RMSE', color='red')
    axs[2].set_title('Root Mean Squared Error over Time Steps')
    axs[2].set_ylabel('RMSE')
    axs[2].legend()

    # R2 Score plot
    axs[3].plot(train_r2, label='Train R2 Score', color='blue')
    axs[3].plot(val_r2, label='Validation R2 Score', color='green')
    axs[3].plot(test_r2, label='Test R2 Score', color='red')
    axs[3].set_title('R2 Score over Time Steps')
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('R2 Score')
    axs[3].legend()

    # Save the plot
    plot_filename = os.path.join(plots_save_dir, f"{index}_{model_type}_error_metrics.jpg")
    plt.savefig(plot_filename)
    plt.show()    
    

# In[2]    

# Get the directory 
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight') 

# Get the directory 
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'datasethourly.xlsx')

dataset = pd.read_excel(file_path, index_col=0)
print(dataset.head())
print(dataset.columns)

# df=dataset[['temp', 'Target','temp_min', 'temp_max', 'pressure',
#         'humidity', 'rain_1h', 'snow_1h',
#         'clouds_all', 'weather_id', 'weather_main', 'weather_description',
#         'day_of_week', 'month', 'season', 'year']]


df=dataset[[ 'Target']]

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df))
dataset = df.values


model_types = ['single_lstm', 'stacked_lstm', 'bidirectional_lstm']

# Load parameters from JSON file
with open('lstm_paramshourly.json', 'r') as file:
    params = json.load(file)
   

# In[3]   

# Access parameters
train_split = params['train_split']
tf_random_seed = params['tf_random_seed']
batch_size = params['batch_size']
learning_rate = params['learning_rate']
future_target = params['future_target']
STEP = params['STEP']
seq_size = params['seq_size']
validation_split = params['validation_split']
epochs = params['epochs']
model_types = params['model_types']
best_params = params['best_params']
# Print parameters
print("train_split:", train_split)
print("tf_random_seed:", tf_random_seed)
print("batch_size:", batch_size)
print("learning_rate:", learning_rate)
print("future_target:", future_target)
print("STEP:", STEP)
print("seq_size:", seq_size)
print("validation_split:", validation_split)
print("epochs:", epochs)
print("model_types:", model_types)
print("best_params:", best_params)

n_splits=6
tf.random.set_seed(73)



# In[4] 


# Ensure the models directory exists
model_save_dir = 'models'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

plots_save_dir = 'plots'
if not os.path.exists(plots_save_dir):
    os.makedirs(plots_save_dir)
    
    

index = 1
feature_index = 0  # Specify which feature/column to plot



splits = TimeSeriesSplit(n_splits)

plt.figure(figsize=(14, 8))

for train_index, test_index in splits.split(dataset):
    train = dataset[train_index]
    test = dataset[test_index]
    
    print(f'Observations: {len(train) + len(test)}')
    print(f'Training Observations: {len(train)}')
    print(f'Testing Observations: {len(test)}')
    print(train.shape)
    print(test.shape)

    
    plt.subplot(n_splits, 1, index)  # Adjust subplot layout based on number of splits
    plt.plot(train[:, feature_index], label='Train')
    plt.plot(np.arange(len(train), len(train) + len(test)), test[:, feature_index], label='Test')
    plt.legend()
    plt.grid(True)
    plt.title(f'Split {index}')
    index += 1
    
plt.tight_layout()
plot_filename = os.path.join(plots_save_dir, f"{n_splits}_Cross Validation.jpg")
plt.savefig(plot_filename)
plt.show()
# Save the plot as a JPG file in the Plots directory with the model name




# In[5]  




feature_index = 0  # Specify which feature/column to plot

model_types=['single_lstm']#, 'stacked_lstm', 'bidirectional_lstm']

for model_type in model_types:
    print(f'Training {model_type}...')
    index = 1
    total_splits = sum(1 for _ in splits.split(df))
    last_split_index = total_splits - 1
    current_split_index = 0
    
    plt.figure(figsize=(14, 8))
    
    for train_index, test_index in splits.split(dataset):
        if current_split_index == last_split_index:

        
            train = dataset[train_index]
            test = dataset[test_index]
        
    
            print(f'Split {index} - Observations: {len(train) + len(test)}')
            print(f'Training Observations: {len(train)}')
            print(train.shape)
            print(f'Testing Observations: {len(test)}')
            print(test.shape)
    
           
            plt.plot(train[:, feature_index], label='Train')
            plt.plot(np.arange(len(train), len(train) + len(test)), test[:, feature_index], label='Test')
            plt.legend()
            plt.title(f'Split {index}')
            plt.grid(True)
            
            plot_filename = os.path.join(plots_save_dir, f"{index}_{model_type}_Cross Validation.jpg")
            plt.savefig(plot_filename)
            plt.show()
            
    
            train_data, val_data, test_data, shape, y_train_multi, y_val_multi, y_test_multi, train_steps, validation_steps, test_steps = prepare_datasets(
            train, test, validation_split, seq_size, future_target, STEP, batch_size,feature_index)
            
    
            model = create_multistep_model(best_params[model_type], model_type, seq_size, future_target, shape)
            model.summary()
            
            best_model, history = train_and_evaluate_multistep_model(
                train_data, val_data, test_data, train_steps, validation_steps, test_steps, model, seq_size, future_target, STEP, epochs)
            
            plot_loss(history, 'Multi-Step Training and validation loss', model_type, index)
            
            model_save_path = os.path.join(model_save_dir, f'best_model_{model_type}_split_{index}.keras')
            best_model.save(model_save_path)
            
            best_model.summary()
            
            # Make predictions using the trained model
            trainPredict = best_model.predict(train_data, steps=train_steps)
            valPredict = best_model.predict(val_data, steps=validation_steps)
            testPredict = best_model.predict(test_data, steps=test_steps)
            
            # Perform further analysis with predictions
            plot_error_metrics_over_time(y_train_multi, trainPredict, y_val_multi, valPredict, y_test_multi, testPredict, future_target,index, model_type)
            
            
    
            
            # Generate plots for predictions on test data
            for i, (x, y) in enumerate(test_data.take(5)):
                    multi_step_plot(x[0], y[0], best_model.predict(x)[0],model_type, index,i)
                
        current_split_index += 1
        index += 1


# In[6]

