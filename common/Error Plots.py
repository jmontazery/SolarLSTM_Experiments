# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:48:51 2024

@author: Jamileh
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns



def plot_r2_for_model(model_name, num_splits=6):
    fig, axs = plt.subplots(num_splits, 1, figsize=(12, 4 * num_splits), sharex=True)
    
    # Initialize an empty list to collect handles and labels for the legend
    handles, labels = [], []

    for split in range(1, num_splits + 1):
        file_name = f"{split}_{model_name}_error_metrics.csv"
        file_path = os.path.join(os.getcwd(), file_name)
        
        if os.path.exists(file_path):
            metrics_df = pd.read_csv(file_path)
            num_timesteps = range(len(metrics_df))
            h1, = axs[split - 1].plot(num_timesteps, metrics_df['Train R2'], label='Train R2', linestyle='-',color='blue')
            h2, = axs[split - 1].plot(num_timesteps, metrics_df['Validation R2'], label='Validation R2', linestyle='-',color='red')
            h3, = axs[split - 1].plot(num_timesteps, metrics_df['Test R2'], label='Test R2', linestyle='-',color='green')
            #axs[split - 1].set_title(f'R2 Score over Time Steps - Split {split}')
            axs[split - 1].set_ylabel(f'R2 Score - Split {split}')
            axs[split - 1].grid(True)
            
            # Collect handles and labels for the legend
            if split == 1:
                handles.extend([h1, h2, h3])
                labels.extend([h1.get_label(), h2.get_label(), h3.get_label()])
        else:
            print(f"File {file_name} not found.")
    
    # Set the x label for the bottom subplot
    axs[-1].set_xlabel('Time Step')

    # Add a single legend below the title
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.91))

    fig.suptitle(f'R2 Scores for {model_name}')
    r2_plot_filename = f'{model_name}_r2_plot.jpg'
    plt.tight_layout(rect=[0, 0, 0.92, 0.92])
    plt.savefig(r2_plot_filename)
    plt.show()



def plot_r2_for_model(model_name, num_splits=6):
    for split in range(1, num_splits + 1):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        file_name = f"{split}_{model_name}_error_metrics.csv"
        file_path = os.path.join(os.getcwd(), file_name)
        
        if os.path.exists(file_path):
            metrics_df = pd.read_csv(file_path)
            num_timesteps = range(len(metrics_df))
            h1, = ax.plot(num_timesteps, metrics_df['Train R2'], label='Train R2', linestyle='-', color='blue')
            h2, = ax.plot(num_timesteps, metrics_df['Validation R2'], label='Validation R2', linestyle='-', color='red')
            h3, = ax.plot(num_timesteps, metrics_df['Test R2'], label='Test R2', linestyle='-', color='green')
            ax.set_ylabel(f'R2 Score - Split {split}')
            ax.set_xlabel('Time Step')
            ax.grid(True)
            
            # Add the legend on the upper left side
            ax.legend(loc='upper right')

            # Add title for each subplot
            #ax.set_title(f'R2 Score over Time Steps - Split {split}')

            # Save each plot separately
            r2_plot_filename = f'{model_name}_r2_plot_split_{split}.jpg'
            plt.tight_layout()
            plt.savefig(r2_plot_filename)
            plt.show()
        else:
            print(f"File {file_name} not found.")


# Example usage
models = ['single_lstm']#, 'stacked_lstm', 'bidirectional_lstm']


# Get the directory 
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight') 
for model in models:
    plot_r2_for_model(model)

