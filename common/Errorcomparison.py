# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:30:25 2024

@author: Jamileh
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


from IPython.display import display  # Import display for Jupyter notebooks

def generate_boxplot_summary_table(combined_data, metric,file_name='summary_table.csv'):
    """
    Generates a summary table with important statistics for the specified metric used in the boxplot.
    The statistics include Median, Q1 (25%), Q3 (75%), IQR, Min, Max, and the count of Outliers.
    
    Parameters:
        combined_data (DataFrame): The data used for creating the boxplot.
        metric (str): The name of the metric column (e.g., 'Test MAE', 'Test RMSE') to summarize.
    
    Returns:
        summary_table (DataFrame): A table with summary statistics for each model.
    """
    summary_table = pd.DataFrame()

    # Calculate summary statistics for each model
    for model in combined_data['Model'].unique():
        model_data = combined_data[combined_data['Model'] == model][metric]
        statistics = {
            'Model': model,
            'Median': model_data.median(),
            'Q1 (25%)': model_data.quantile(0.25),
            'Q3 (75%)': model_data.quantile(0.75),
            'IQR': model_data.quantile(0.75) - model_data.quantile(0.25),
            'Min': model_data.min(),
            'Max': model_data.max(),
            'Outliers': ((model_data < model_data.quantile(0.25) - 1.5 * (model_data.quantile(0.75) - model_data.quantile(0.25))) | 
                         (model_data > model_data.quantile(0.75) + 1.5 * (model_data.quantile(0.75) - model_data.quantile(0.25)))).sum()
        }
        # Use pd.concat to add the statistics row to the summary_table
        summary_table = pd.concat([summary_table, pd.DataFrame([statistics])], ignore_index=True)

    summary_table = summary_table.sort_values('Median')  # Optional: Sort models by Median for clarity

    # Display the summary table using standard display functions
    display(summary_table)
    summary_table.to_csv(file_name, index=False)

    return summary_table

# Function to read the R² error data from files
def read_r2_data(file_path):
    """
    Reads the R² error data from a file.
    Assumes each file contains the future target index and the corresponding R² error.
    """
    return pd.read_csv(file_path)

# Get a list of all CSV files in the folder
def get_csv_files(folder_path):
    """
    Returns a list of all CSV file paths in the given folder.
    """
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

def plot_r2_data(csv_files, selected_files):
    """
    Plots the R² data from the selected CSV files in black and white with diversified line styles and shades of gray.
    """
    plt.figure(figsize=(14, 6))
    
    # Extended and diversified line styles for up to 10 different files
    # line_styles = [
    #     '-', '--', '-.', ':', 
    #     (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (5, 2, 5, 2)), 
    #     (0, (3, 1, 1, 1, 1, 1)), (0, (1, 2))
    # ]
    
    # # Shades of gray for up to 10 different files (made more distinct)
    # colors = [
    #     'black', 'dimgray', 'gray', 'darkgray', 'silver', 
    #     'lightgray', 'gainsboro', 'slategray', 'dimgrey', 'darkslategray'
    # ]
    
    # Line styles
    line_styles = [
        '-', '--', '-.', ':', 
        (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (5, 2, 5, 2)), 
        (0, (3, 1, 1, 1, 1, 1))
    ]
    
    # Shades of gray for distinct colors
    colors = [
        'black', 'dimgray', 'gray', 'darkgray', 'silver', 
        'lightgray', 'gainsboro', 'slategray', 'dimgrey'
    ]
    
    # Markers for clarity
    markers = ['','','o', 's', 'D', '^', 'v', 'p', '*', 'h', 'x']


    for idx, file_path in enumerate(csv_files):
        file_name = os.path.basename(file_path)
        if file_name in selected_files:
            data = read_r2_data(file_path)
            label = file_name.replace('.csv', '')
            plt.plot(data['Time Step'], data['Test R2'], 
                     label=label, linestyle=line_styles[idx % len(line_styles)], 
                     marker=markers[idx % len(markers)],            # Markers for data points
                     markersize=5,                   
                     color=colors[idx % len(colors)], linewidth=2 if idx > 4 else 1.5)  # Thicker lines for lighter colors
    
    
    # # Add vertical lines at hour 6 and hour 18 to mark sections
    # plt.axvline(x=6, color='black', linestyle='--', linewidth=1)
    # plt.axvline(x=12, color='black', linestyle='--', linewidth=1)
    # plt.axvline(x=18, color='black', linestyle='--', linewidth=1)

    
    # Add vertical lines at hour 6 and hour 18 to mark sections for quarter hourly
    plt.axvline(x=24, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=48, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=72, color='black', linestyle='--', linewidth=1)

    # Customize the plot
    #plt.title('Comparison of R² for Different LSTM Models (Quarter-Hourly Data)', fontsize=14)
    plt.xlabel('Future Target (Time Step)', fontsize=12)
    plt.ylabel('R²', fontsize=12)
    #plt.ylim(0.88, 1)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle=':', linewidth=0.5, color='lightgray', which='both')  # Lighter grid
    plt.minorticks_on()
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Save the plot as a high-quality image
    plt.savefig('lstm_model_comparison_plot.jpg', dpi=300, bbox_inches='tight')




# Function to plot side-by-side comparison of hourly vs quarterly data
def plot_side_by_side_comparison(csv_files, selected_hourly, selected_quarterly):
    """
    Plots side-by-side boxplots for hourly and quarterly data to compare error (e.g., R²) distribution across models.
    """
    # Create empty DataFrames to store combined data for hourly and quarterly models
    combined_hourly_data = pd.DataFrame()
    combined_quarterly_data = pd.DataFrame()
    
    # Combine data for hourly models
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if file_name in selected_hourly:
            data = read_r2_data(file_path)
            data['Model'] = file_name.replace('.csv', '')  # Add a 'Model' column to label each model
            combined_hourly_data = pd.concat([combined_hourly_data, data], axis=0)
    
    # Combine data for quarterly models
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if file_name in selected_quarterly:
            data = read_r2_data(file_path)
            data['Model'] = file_name.replace('.csv', '')  # Add a 'Model' column to label each model
            combined_quarterly_data = pd.concat([combined_quarterly_data, data], axis=0)
    
    # Set up the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)  # Side-by-side plots
    
    # Hourly boxplot
    sns.boxplot(ax=axes[0], x='Model', y='Test R2', data=combined_hourly_data, palette="Greys", linewidth=1.2)
    axes[0].set_title(' Hourly Data', fontsize=14)
    axes[0].set_xlabel('', fontsize=12)
    axes[0].set_ylabel('R²', fontsize=12)
    axes[0].grid(True, linestyle=':', linewidth=0.5, color='black')
    axes[0].tick_params(axis='x', rotation=45, labelsize=10)  # Rotate x labels for better readability
    
    # Quarterly boxplot
    sns.boxplot(ax=axes[1], x='Model', y='Test R2', data=combined_quarterly_data, palette="Greys", linewidth=1.2)
    axes[1].set_title('Quarter hourly Data', fontsize=14)
    axes[1].set_xlabel('', fontsize=12)
    axes[1].set_ylabel('', fontsize=12)
    axes[1].grid(True, linestyle=':', linewidth=0.5, color='black')
    axes[1].tick_params(axis='x', rotation=45, labelsize=10)
    
    # Adjust y-axis limits if needed for better comparison
    axes[0].set_ylim(0.74, 1.0)  # Set y-axis limits to focus on a specific range
    axes[1].set_ylim(0.74, 1.0)  # Ensure both plots share the same y-axis range
    
    plt.tight_layout()
    plt.show()
    
    # Optionally, save the plot as a high-quality image
    plt.savefig('lstm_error_distribution_comparison_bw.jpg', dpi=300, bbox_inches='tight')




# Function to read the R² error data and calculate the mean and std across all steps
def calculate_mean_and_std(file, time_conversion_factor=1):
    """
    Reads a file and calculates the mean and standard deviation (std) of R² values across all time steps.
    For quarter-hourly data, the time_conversion_factor will be 4 (to convert time steps to hours).
    
    Parameters:
    - file: The CSV file to be read.
    - time_conversion_factor: Conversion factor to adjust time steps (for quarter-hourly data).
    
    Returns:
    - A tuple of (mean_r2, std_r2) for the file.
    """
    # Read the file
    data = pd.read_csv(file)
    
    # Convert time steps to hours if necessary
    time_steps = data['Time Step'] / time_conversion_factor
    r2_errors = data['Test R2']
    
    # Calculate mean and std of R² across all time steps
    mean_r2 = r2_errors.mean()
    std_r2 = r2_errors.std()
    
    return mean_r2, std_r2

# Function to calculate mean R² errors for 6-hour intervals
def calculate_mean_error_intervals(file, time_conversion_factor=1):
    """
    Reads a file and calculates the mean R² error for each 6-hour interval.
    
    Parameters:
    - file: The CSV file to be read.
    - time_conversion_factor: Conversion factor to adjust time steps (for quarter-hourly data).
    
    Returns:
    - A list of mean R² errors for the 6-hour intervals.
    """
    # Read the file
    data = pd.read_csv(file)
    
    # Convert time steps to hours if necessary
    time_steps = data['Time Step'] / time_conversion_factor
    r2_errors = data['Test R2']
    
    # Define the intervals (0-6 hours, 6-12 hours, 12-18 hours, 18-24 hours)
    intervals = [(0, 6), (6, 12), (12, 18), (18, 24)]
    mean_errors = []
    
    # Calculate mean R² error for each interval
    for start, end in intervals:
        mask = (time_steps >= start) & (time_steps < end)
        mean_r2 = r2_errors[mask].mean()  # Calculate mean for the interval
        mean_errors.append(mean_r2)
    
    return mean_errors

# Function to combine results and save as CSV, including mean and std
def summarize_and_save_mean_errors_combined(hourly_files, quarter_hourly_files, selected_files_hourly, selected_files_quarter_hourly, output_file='combined_mean_errors.csv'):
    """
    Summarizes the mean R² errors for hourly and quarter-hourly files, including the overall mean and standard deviation (std),
    and saves them all in one CSV file.
    """
    # Initialize a list to collect rows for the final table
    all_model_errors = []
    
    # Column names for the intervals
    columns = ['0-6 hours', '6-12 hours', '12-18 hours', '18-24 hours', 'Mean R²', 'Std R²']
    
    # Process hourly models
    for file_path in hourly_files:
        file_name = os.path.basename(file_path)
        if file_name in selected_files_hourly:
            mean_errors = calculate_mean_error_intervals(file_path, time_conversion_factor=1)
            mean_r2, std_r2 = calculate_mean_and_std(file_path, time_conversion_factor=1)
            all_model_errors.append([file_name.replace('.csv', '')] + mean_errors + [mean_r2, std_r2])

    # Process quarter-hourly models
    for file_path in quarter_hourly_files:
        file_name = os.path.basename(file_path)
        if file_name in selected_files_quarter_hourly:
            mean_errors = calculate_mean_error_intervals(file_path, time_conversion_factor=4)
            mean_r2, std_r2 = calculate_mean_and_std(file_path, time_conversion_factor=4)
            all_model_errors.append([file_name.replace('.csv', '')] + mean_errors + [mean_r2, std_r2])
    
    # Create a DataFrame from the collected results
    df_combined = pd.DataFrame(all_model_errors, columns=['Model'] + columns)
    df_combined.set_index('Model', inplace=True)  # Set the 'Model' column as the row index
    
    # Save the combined result as a CSV file
    df_combined.to_csv(output_file)
    print(f"Combined results saved to: {output_file}")




# Function to plot boxplot showing MAE distribution across models in black and white
def plot_MAE_distribution(csv_files, selected_files):
    """
    Plots the MAE distribution across different models as a boxplot in black and white, 
    arranged in the order of the filenames provided in selected_files.
    """
    combined_data = pd.DataFrame()  # Empty dataframe to store combined data from all models
    
    # Loop through each file and extract MAE data
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if file_name in selected_files:
            data = read_r2_data(file_path)
            data['Model'] = file_name.replace('.csv', '')  # Add a 'Model' column to label each model
            combined_data = pd.concat([combined_data, data], axis=0)  # Append to combined dataframe
    
    # Ensure that the 'Model' column follows the order of selected_files
    combined_data['Model'] = pd.Categorical(
        combined_data['Model'], 
        categories=[file.replace('.csv', '') for file in selected_files], 
        ordered=True
    )
    
    # Create a boxplot in black and white to show the MAE distribution across models
    plt.figure(figsize=(14, 6))
    sns.set(style="whitegrid")  # Set the background to white for better contrast
    sns.boxplot(x='Model', y='Test MAE', data=combined_data, palette="Greys", linewidth=1.2)
    generate_boxplot_summary_table(combined_data, 'Test MAE')
    # Customize the plot
    plt.title('MAE Distribution Across Different LSTM Models', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.grid(True, linestyle=':', linewidth=0.5, color='black')  # Black gridlines for better visibility
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability
    plt.tight_layout()
    
    # Show the plot
    plt.show()

    # Optionally, save the plot as a high-quality image
    plt.savefig('lstm_mae_distribution_boxplot_bw.jpg', dpi=300, bbox_inches='tight')


# Function to plot boxplot showing MAE distribution across models in black and white
def plot_MSE_distribution(csv_files, selected_files):
    """
    Plots the MSE distribution across different models as a boxplot in black and white, 
    arranged in the order of the filenames provided in selected_files.
    """
    combined_data = pd.DataFrame()  # Empty dataframe to store combined data from all models
    
    # Loop through each file and extract MSE data
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if file_name in selected_files:
            data = read_r2_data(file_path)
            data['Model'] = file_name.replace('.csv', '')  # Add a 'Model' column to label each model
            combined_data = pd.concat([combined_data, data], axis=0)  # Append to combined dataframe
    
    # Ensure that the 'Model' column follows the order of selected_files
    combined_data['Model'] = pd.Categorical(
        combined_data['Model'], 
        categories=[file.replace('.csv', '') for file in selected_files], 
        ordered=True
    )
    
    # Create a boxplot in black and white to show the MSE distribution across models
    plt.figure(figsize=(14, 6))
    sns.set(style="whitegrid")  # Set the background to white for better contrast
    sns.boxplot(x='Model', y='Test MSE', data=combined_data, palette="Greys", linewidth=1.2)

    # Customize the plot
    plt.title('MSE Distribution Across Different LSTM Models', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.grid(True, linestyle=':', linewidth=0.5, color='black')  # Black gridlines for better visibility
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability
    plt.tight_layout()
    
    # Show the plot
    plt.show()

    # Optionally, save the plot as a high-quality image
    plt.savefig('lstm_MSE_distribution_boxplot_bw.jpg', dpi=300, bbox_inches='tight')


# Function to plot boxplot showing RMSE distribution across models in black and white
def plot_RMSE_distribution(csv_files, selected_files):
    """
    Plots the RMSE distribution across different models as a boxplot in black and white, 
    arranged in the order of the filenames provided in selected_files.
    """
    combined_data = pd.DataFrame()  # Empty dataframe to store combined data from all models
    
    # Loop through each file and extract RMSE data
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if file_name in selected_files:
            data = read_r2_data(file_path)
            data['Model'] = file_name.replace('.csv', '')  # Add a 'Model' column to label each model
            combined_data = pd.concat([combined_data, data], axis=0)  # Append to combined dataframe
    
    # Ensure that the 'Model' column follows the order of selected_files
    combined_data['Model'] = pd.Categorical(
        combined_data['Model'], 
        categories=[file.replace('.csv', '') for file in selected_files], 
        ordered=True
    )
    
    # Create a boxplot in black and white to show the RMSE distribution across models
    plt.figure(figsize=(14, 6))
    sns.set(style="whitegrid")  # Set the background to white for better contrast
    sns.boxplot(x='Model', y='Test RMSE', data=combined_data, palette="Greys", linewidth=1.2)

    # Customize the plot
    plt.title('RMSE Distribution Across Different LSTM Models', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True, linestyle=':', linewidth=0.5, color='black')  # Black gridlines for better visibility
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability
    plt.tight_layout()
    
    # Show the plot
    plt.show()

    # Optionally, save the plot as a high-quality image
    plt.savefig('lstm_RMSE_distribution_boxplot_bw.jpg', dpi=300, bbox_inches='tight')


# Function to plot boxplot showing MAE distribution across models in black and white
def plot_R2_distribution(csv_files, selected_files):
    """
    Plots the R2 distribution across different models as a boxplot in black and white, 
    arranged in the order of the filenames provided in selected_files.
    """
    combined_data = pd.DataFrame()  # Empty dataframe to store combined data from all models
    
    # Loop through each file and extract R2 data
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if file_name in selected_files:
            data = read_r2_data(file_path)
            data['Model'] = file_name.replace('.csv', '')  # Add a 'Model' column to label each model
            combined_data = pd.concat([combined_data, data], axis=0)  # Append to combined dataframe
    
    # Ensure that the 'Model' column follows the order of selected_files
    combined_data['Model'] = pd.Categorical(
        combined_data['Model'], 
        categories=[file.replace('.csv', '') for file in selected_files], 
        ordered=True
    )
    
    # Create a boxplot in black and white to show the R2 distribution across models
    plt.figure(figsize=(14, 6))
    sns.set(style="whitegrid")  # Set the background to white for better contrast
    sns.boxplot(x='Model', y='Test R2', data=combined_data, palette="Greys", linewidth=1.2)

    # Customize the plot
    plt.title('R2 Distribution Across Different LSTM Models', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('R2', fontsize=12)
    plt.grid(True, linestyle=':', linewidth=0.5, color='black')  # Black gridlines for better visibility
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability
    plt.tight_layout()
    
    # Show the plot
    plt.show()

    # Optionally, save the plot as a high-quality image
    plt.savefig('lstm_R2_distribution_boxplot_bw.jpg', dpi=300, bbox_inches='tight')


def plot_r2_dataH(csv_files, selected_files):
    """
    Plots the R² data from the selected CSV files in black and white with diversified line styles and shades of gray.
    """
    plt.figure(figsize=(14, 6))

    # Predefined style map to ensure consistent styles across hourly and quarter-hourly datasets
    style_map = {
        'LSTMHUni': {'linestyle': '-', 'color': 'black', 'marker': 'o'},
        'LSTMHT': {'linestyle': '--', 'color': 'dimgray', 'marker': 's'},
        'LSTMHTC': {'linestyle': '-.', 'color': 'gray', 'marker': 'D'},
        'BiHUni': {'linestyle': ':', 'color': 'darkgray', 'marker': '^'},
        'BiHT': {'linestyle': (0, (3, 1, 1, 1)), 'color': 'black', 'marker': 'v'},
        'BiHTC': {'linestyle': (0, (1, 1)), 'color': 'dimgray', 'marker': 'P'},
        'StackedHUni': {'linestyle': (0, (5, 1)), 'color': 'silver', 'marker': 'X'},
        'StackedHT': {'linestyle': (0, (3, 1, 1, 1, 1, 1)), 'color': 'darkgray', 'marker': '*'},
        'StackedHTC': {'linestyle': (0, (5, 2, 5, 2)), 'color': 'lightgray', 'marker': '+'},
    
        # Quarter-hourly models with the same styles as their hourly counterparts
        'LSTMQUni': {'linestyle': '-', 'color': 'black', 'marker': 'o'},
        'LSTMQT': {'linestyle': '--', 'color': 'dimgray', 'marker': 's'},
        'LSTMQTC': {'linestyle': '-.', 'color': 'gray', 'marker': 'D'},
        'BiQUni': {'linestyle': ':', 'color': 'darkgray', 'marker': '^'},
        'BiQT': {'linestyle': (0, (3, 1, 1, 1)), 'color': 'black', 'marker': 'v'},
        'BiQTC': {'linestyle': (0, (1, 1)), 'color': 'dimgray', 'marker': 'P'},
        'StackedQUni': {'linestyle': (0, (5, 1)), 'color': 'silver', 'marker': 'X'},
        'StackedQT': {'linestyle': (0, (3, 1, 1, 1, 1, 1)), 'color': 'darkgray', 'marker': '*'},
        'StackedQTC': {'linestyle': (0, (5, 2, 5, 2)), 'color': 'lightgray', 'marker': '+'},
    }

    for idx, file_path in enumerate(csv_files):
        file_name = os.path.basename(file_path)
        if file_name in selected_files:
            data = read_r2_data(file_path)
            label = file_name.replace('.csv', '')
            style = style_map.get(label, {'linestyle': '-', 'color': 'black', 'marker': ''})
            plt.plot(data['Time Step'], data['Test R2'], 
                     label=label, linestyle=style['linestyle'], 
                     marker=style['marker'], color=style['color'], linewidth=2 if idx > 4 else 1.5)


    # Add vertical lines for quarter-hourly data
    plt.axvline(x=6, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=12, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=18, color='black', linestyle='--', linewidth=1)
    # Customize the plot
    #plt.title('Comparison of R² for Different LSTM Models (Quarter-Hourly Data)', fontsize=14)
    plt.xlabel('Future Target (Time Step)', fontsize=12)
    plt.ylabel('R²', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle=':', linewidth=0.5, color='lightgray', which='both')
    plt.minorticks_on()
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Save the plot as a high-quality image
    plt.savefig('lstm_model_comparison_plot.jpg', dpi=300, bbox_inches='tight')


def plot_r2_dataQ(csv_files, selected_files):
    """
    Plots the R² data from the selected CSV files in black and white with diversified line styles and reduced marker frequency.
    """
    plt.figure(figsize=(14, 6))

    # Predefined style map with simplified line styles and diverse markers
    style_map = {
        'LSTMHUni': {'linestyle': '-', 'color': 'black', 'marker': 'o'},
        'LSTMHT': {'linestyle': '--', 'color': 'dimgray', 'marker': 's'},
        'LSTMHTC': {'linestyle': '-.', 'color': 'gray', 'marker': 'D'},
        'BiHUni': {'linestyle': ':', 'color': 'darkgray', 'marker': '^'},
        'BiHT': {'linestyle': '-', 'color': 'black', 'marker': 'v'},
        'BiHTC': {'linestyle': '--', 'color': 'dimgray', 'marker': 'P'},
        'StackedHUni': {'linestyle': '-.', 'color': 'silver', 'marker': 'X'},
        'StackedHT': {'linestyle': ':', 'color': 'darkgray', 'marker': '*'},
        'StackedHTC': {'linestyle': '-', 'color': 'lightgray', 'marker': '+'},

        # Quarter-hourly models with the same styles as their hourly counterparts
        'LSTMQUni': {'linestyle': '-', 'color': 'black', 'marker': 'o'},
        'LSTMQT': {'linestyle': '--', 'color': 'dimgray', 'marker': 's'},
        'LSTMQTC': {'linestyle': '-.', 'color': 'gray', 'marker': 'D'},
        'BiQUni': {'linestyle': ':', 'color': 'darkgray', 'marker': '^'},
        'BiQT': {'linestyle': '-', 'color': 'black', 'marker': 'v'},
        'BiQTC': {'linestyle': '--', 'color': 'dimgray', 'marker': 'P'},
        'StackedQUni': {'linestyle': '-.', 'color': 'silver', 'marker': 'X'},
        'StackedQT': {'linestyle': ':', 'color': 'darkgray', 'marker': '*'},
        'StackedQTC': {'linestyle': '-', 'color': 'lightgray', 'marker': '+'},
    }

    for idx, file_path in enumerate(csv_files):
        file_name = os.path.basename(file_path)
        if file_name in selected_files:
            data = read_r2_data(file_path)
            label = file_name.replace('.csv', '')
            style = style_map.get(label, {'linestyle': '-', 'color': 'black', 'marker': ''})

            # Plot the line without markers first
            plt.plot(data['Time Step'], data['Test R2'], 
                     label=label, linestyle=style['linestyle'], 
                     marker=style['marker'],markevery=4,
                     markersize=6, alpha=0.8,
                     color=style['color'], linewidth=2 if idx > 4 else 1.5)

            # # Add markers only at every 10th point
            # plt.plot(data['Time Step'][::4], data['Test R2'][::4], 
            #          linestyle='None', marker=style['marker'], 
            #          markersize=6, color=style['color'], alpha=0.8)  # Adding transparency to markers

    # Add vertical lines for quarter-hourly data
    plt.axvline(x=24, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=48, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=72, color='black', linestyle='--', linewidth=1)

    # Customize the plot
    #plt.title('Comparison of R² for Different LSTM Models (Quarter-Hourly Data)', fontsize=14)
    plt.xlabel('Future Target (Time Step)', fontsize=12)
    plt.ylabel('R²', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle=':', linewidth=0.5, color='lightgray', which='both')
    plt.minorticks_on()
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Save the plot as a high-quality image
    plt.savefig('lstm_model_comparison_plot.jpg', dpi=300, bbox_inches='tight')





# Folder path where the CSV files are located
folder_path = os.getcwd()# Get the list of all CSV files in the folder
csv_files = get_csv_files(folder_path)


LSTMH = ['LSTMHUni.csv', 'LSTMHT.csv','LSTMHTC.csv'] 
StackedH = [ 'StackedHUni.csv','StackedHT.csv','StackedHTC.csv']
BiH = ['BiHUni.csv', 'BiHT.csv', 'BiHTC.csv']


UNIH = ['LSTMHUni.csv', 'StackedHUni.csv','BiHUni.csv']
ONEH = ['LSTMHT.csv','StackedHT.csv', 'BiHT.csv']
FIVEH = ['LSTMHTC.csv','StackedHTC.csv', 'BiHTC.csv']



LSTMQ = ['LSTMQUni.csv', 'LSTMQT.csv','LSTMQTC.csv'] 
StackedQ = [ 'StackedQUni.csv','StackedQT.csv','StackedQTC.csv']
BiQ = ['BiQUni.csv', 'BiQT.csv', 'BiQTC.csv']


UNIQ = ['LSTMQUni.csv', 'StackedQUni.csv','BiQUni.csv']
ONEQ = ['LSTMQT.csv','StackedQT.csv', 'BiQT.csv']
FIVEQ = ['LSTMQTC.csv','StackedQTC.csv', 'BiQTC.csv']


all_models=['LSTMHUni.csv',
'LSTMQUni.csv',
'LSTMHT.csv',
'LSTMQT.csv', 
'LSTMHTC.csv', 
'LSTMQTC.csv', 
'BiHUni.csv', 
'BiQUni.csv', 
'BiHT.csv', 
'BiQT.csv',
'BiHTC.csv', 
'BiQTC.csv',
'StackedHUni.csv', 
'StackedQUni.csv', 
'StackedHT.csv',
'StackedQT.csv',
'StackedHTC.csv',
'StackedQTC.csv']


selected_hourly = ['LSTMHUni.csv',
'LSTMHT.csv',
'LSTMHTC.csv', 
'BiHUni.csv', 
'BiHT.csv', 
'BiHTC.csv', 
'StackedHUni.csv', 
'StackedHT.csv',
'StackedHTC.csv']


selected_Qhourly = ['LSTMQUni.csv',
'LSTMQT.csv', 
'LSTMQTC.csv', 
'BiQUni.csv', 
'BiQT.csv',
'BiQTC.csv',
'StackedQUni.csv', 
'StackedQT.csv',
'StackedQTC.csv']


all_modelsHQ=['LSTMHUni.csv',
'LSTMHT.csv',
'LSTMHTC.csv', 
'BiHUni.csv', 
'BiHT.csv', 
'BiHTC.csv', 
'StackedHUni.csv', 
'StackedHT.csv',
'StackedHTC.csv',
'LSTMQUni.csv',
'LSTMQT.csv', 
'LSTMQTC.csv', 
'BiQUni.csv', 
'BiQT.csv',
'BiQTC.csv',
'StackedQUni.csv', 
'StackedQT.csv',
'StackedQTC.csv']




all_modelsTC=[
'LSTMHTC.csv', 
'BiHTC.csv', 
'StackedHTC.csv',
'LSTMQTC.csv', 
'BiQTC.csv',
'StackedQTC.csv']

all_modelsT=[
'LSTMHT.csv',
'BiHT.csv', 
'StackedHT.csv',
'LSTMQT.csv', 
'BiQT.csv',
'StackedQT.csv']

all_modelsUNI=['LSTMHUni.csv',
'BiHUni.csv', 
'StackedHUni.csv', 
'LSTMQUni.csv',
'BiQUni.csv', 
'StackedQUni.csv']

# Plot the side-by-side comparison for hourly and quarterly data
plot_side_by_side_comparison(csv_files, selected_hourly, selected_Qhourly)


# Save the summarized results
summarize_and_save_mean_errors_combined(csv_files, csv_files, selected_hourly, selected_Qhourly)



# Plot the R² data for the selected files
plot_r2_dataQ(csv_files, selected_Qhourly)


# Plot the R² data for the selected files
plot_r2_dataH(csv_files, selected_hourly)



# Plot the error distribution for selected quarterly files

plot_MSE_distribution(csv_files, all_modelsHQ)

plot_MAE_distribution(csv_files, all_modelsHQ)

plot_R2_distribution(csv_files, all_modelsHQ)

plot_RMSE_distribution(csv_files, all_modelsHQ)




# Function to plot side-by-side comparison of univariate, multivariate with T, and multivariate with TC models
def plot_univariate_multivariate_comparison(csv_files, all_modelsUNI, all_modelsT, all_modelsTC):
    """
    Plots side-by-side boxplots for univariate models, multivariate models with temperature (T), 
    and multivariate models with both temperature and calendar (TC) data to compare RMSE distribution across models.
    """
    # Create empty DataFrames to store combined data for each category
    combined_uni_data = pd.DataFrame()
    combined_t_data = pd.DataFrame()
    combined_tc_data = pd.DataFrame()
    
    # Combine data for univariate models
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if file_name in all_modelsUNI:
            data = read_r2_data(file_path)
            data['Model'] = file_name.replace('.csv', '')  # Add a 'Model' column to label each model
            combined_uni_data = pd.concat([combined_uni_data, data], axis=0)
    
    # Combine data for multivariate models with temperature (T)
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if file_name in all_modelsT:
            data = read_r2_data(file_path)
            data['Model'] = file_name.replace('.csv', '')  # Add a 'Model' column to label each model
            combined_t_data = pd.concat([combined_t_data, data], axis=0)
    
    # Combine data for multivariate models with temperature and calendar (TC)
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if file_name in all_modelsTC:
            data = read_r2_data(file_path)
            data['Model'] = file_name.replace('.csv', '')  # Add a 'Model' column to label each model
            combined_tc_data = pd.concat([combined_tc_data, data], axis=0)
    
    # Set up the figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)  # Side-by-side plots for the three categories
    
    # Univariate boxplot
    sns.boxplot(ax=axes[0], x='Model', y='Test RMSE', data=combined_uni_data, color='lightgray', linewidth=1.2)
    axes[0].set_title('Univariate Models', fontsize=14)
    #axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('RMSE', fontsize=12)
    axes[0].grid(True, linestyle=':', linewidth=0.5, color='black')
    axes[0].tick_params(axis='x', rotation=45, labelsize=10)  # Rotate x labels for better readability
    
    # Multivariate with Temperature (T) boxplot
    sns.boxplot(ax=axes[1], x='Model', y='Test RMSE', data=combined_t_data, color='gray', linewidth=1.2)
    axes[1].set_title('Multivariate (T) Models', fontsize=14)
    #axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('', fontsize=12)
    axes[1].grid(True, linestyle=':', linewidth=0.5, color='black')
    axes[1].tick_params(axis='x', rotation=45, labelsize=10)
    
    # Multivariate with Temperature and Calendar (TC) boxplot
    sns.boxplot(ax=axes[2], x='Model', y='Test RMSE', data=combined_tc_data, color='black', linewidth=1.2)
    axes[2].set_title('Multivariate (TC) Models', fontsize=14)
    #axes[2].set_xlabel('Model', fontsize=12)
    axes[2].set_ylabel('', fontsize=12)
    axes[2].grid(True, linestyle=':', linewidth=0.5, color='black')
    axes[2].tick_params(axis='x', rotation=45, labelsize=10)
    
    # Adjust y-axis limits if needed for better comparison across the subplots
    axes[0].set_ylim(0, 0.12)  # Set y-axis limits to focus on a specific range for better visualization
    axes[1].set_ylim(0, 0.12)
    axes[2].set_ylim(0, 0.12)
    
    plt.tight_layout()
    plt.show()
    
    # Optionally, save the plot as a high-quality image
    plt.savefig('rmse_comparison_univariate_multivariate.jpg', dpi=300, bbox_inches='tight')

# Call the function with your specific model groups
plot_univariate_multivariate_comparison(csv_files, all_modelsUNI, all_modelsT, all_modelsTC)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Function to plot side-by-side comparison for any error metric
def plot_error_distribution_comparison(csv_files, all_modelsUNI, all_modelsT, all_modelsTC, error_metric, metric_name, y_limit=None):
    """
    Plots side-by-side boxplots for univariate models, multivariate models with temperature (T), 
    and multivariate models with both temperature and calendar (TC) data to compare the distribution of error metrics.
    
    Args:
        error_metric: The column name in the data to use for plotting (e.g., 'Test RMSE', 'Test MAE', 'Test MSE').
        metric_name: The name of the error metric for labeling the plots (e.g., 'RMSE', 'MAE', 'MSE').
        y_limit: Optional parameter to specify y-axis limits for the plot.
    """
    # Create empty DataFrames to store combined data for each category
    combined_uni_data = pd.DataFrame()
    combined_t_data = pd.DataFrame()
    combined_tc_data = pd.DataFrame()
    
    # Combine data for univariate models
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if file_name in all_modelsUNI:
            data = read_r2_data(file_path)
            data['Model'] = file_name.replace('.csv', '')  # Add a 'Model' column to label each model
            combined_uni_data = pd.concat([combined_uni_data, data], axis=0)
    
    # Combine data for multivariate models with temperature (T)
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if file_name in all_modelsT:
            data = read_r2_data(file_path)
            data['Model'] = file_name.replace('.csv', '')  # Add a 'Model' column to label each model
            combined_t_data = pd.concat([combined_t_data, data], axis=0)
    
    # Combine data for multivariate models with temperature and calendar (TC)
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if file_name in all_modelsTC:
            data = read_r2_data(file_path)
            data['Model'] = file_name.replace('.csv', '')  # Add a 'Model' column to label each model
            combined_tc_data = pd.concat([combined_tc_data, data], axis=0)
    
    # Set up the figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)  # Side-by-side plots for the three categories
    
    # Univariate boxplot
    sns.boxplot(ax=axes[0], x='Model', y=error_metric, data=combined_uni_data, color='lightgray', linewidth=1.2)
    axes[0].set_title('Univariate Models', fontsize=14)
    axes[0].set_xlabel('')  # Remove the x-axis label "Model"
    axes[0].set_ylabel(metric_name, fontsize=12)
    axes[0].grid(True, linestyle=':', linewidth=0.5, color='black')
    axes[0].tick_params(axis='x', rotation=45, labelsize=10)  # Rotate x labels for better readability
    
    # Multivariate with Temperature (T) boxplot
    sns.boxplot(ax=axes[1], x='Model', y=error_metric, data=combined_t_data, color='gray', linewidth=1.2)
    axes[1].set_title('Multivariate (T) Models', fontsize=14)
    axes[1].set_xlabel('')  # Remove the x-axis label "Model"
    axes[1].set_ylabel('', fontsize=12)
    axes[1].grid(True, linestyle=':', linewidth=0.5, color='black')
    axes[1].tick_params(axis='x', rotation=45, labelsize=10)
    
    # Multivariate with Temperature and Calendar (TC) boxplot
    sns.boxplot(ax=axes[2], x='Model', y=error_metric, data=combined_tc_data, color='black', linewidth=1.2)
    axes[2].set_title('Multivariate (TC) Models', fontsize=14)
    axes[2].set_xlabel('')  # Remove the x-axis label "Model"
    axes[2].set_ylabel('', fontsize=12)
    axes[2].grid(True, linestyle=':', linewidth=0.5, color='black')
    axes[2].tick_params(axis='x', rotation=45, labelsize=10)
    
    # Adjust y-axis limits if provided for better comparison across the subplots
    if y_limit:
        axes[0].set_ylim(0, y_limit)
        axes[1].set_ylim(0, y_limit)
        axes[2].set_ylim(0, y_limit)
    
    plt.tight_layout()
    plt.show()
    
    # Optionally, save the plot as a high-quality image
    plt.savefig(f'{metric_name.lower()}_comparison_univariate_multivariate.jpg', dpi=300, bbox_inches='tight')

# Plot the comparison for RMSE with a suitable y-limit
plot_error_distribution_comparison(csv_files, all_modelsUNI, all_modelsT, all_modelsTC, 'Test RMSE', 'RMSE', y_limit=0.14)

# Plot the comparison for MAE with a different y-limit
plot_error_distribution_comparison(csv_files, all_modelsUNI, all_modelsT, all_modelsTC, 'Test MAE', 'MAE', y_limit=0.08)

# Plot the comparison for MSE with a different y-limit
plot_error_distribution_comparison(csv_files, all_modelsUNI, all_modelsT, all_modelsTC, 'Test MSE', 'MSE', y_limit=0.017)
