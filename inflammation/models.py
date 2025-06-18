"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np
import json
import os
import glob

class CSVDataSource:
    '''Class for generating data from inflammation*.csv files.'''
    def __init__(self, dir_path):
        self.dir_path = dir_path
    
    def load_inflammation_data(self):
        '''Returns list of 2D arrays from inflammation*.csv files.'''
        data_file_paths = glob.glob(os.path.join(self.dir_path, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation CSV files found in path {self.dir_path}")
        data = map(load_csv, data_file_paths) # Load inflammation data from each CSV file
        return list(data) # Return the list of 2D NumPy arrays with inflammation data
    
class JSONDataSource:
    '''Class for generating data from inflammation*.json files.'''
    def __init__(self, dir_path):
        self.dir_path = dir_path
    
    def load_inflammation_data(self):
        '''Returns list of 2D arrays from inflammation*.json files.'''
        data_file_paths = glob.glob(os.path.join(self.dir_path, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation JSON files found in path {self.dir_path}")
        data = map(load_json, data_file_paths) # Load inflammation data from each JSON file
        return list(data) # Return the list of 2D NumPy arrays with inflammation data


def load_csv(filename):  
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')

def load_json(filename):
    """Load a numpy array from a JSON document.
    
    Expected format:
    [
      {
        "observations": [0, 1]
      },
      {
        "observations": [0, 2]
      }    
    ]
    :param filename: Filename of CSV to load
    """
    with open(filename, 'r', encoding='utf-8') as file:
        data_as_json = json.load(file)
        return [np.array(entry['observations']) for entry in data_as_json]


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array for each day.

   :param data: A 2D data array with inflammation data (each row contains measurements
             for a single patient across all days).
   :returns: An array of mean values of measurements for each day.
   """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.
    
    :param data: A 2D data array with inflammation data (each row contains measurements 
            for a single patient across all days).
    :returns: An array of maximum values of the measurement for each day.
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.
    
    :param data: A 2D data array with inflammation data (each row contains measurements 
            for a single patient across all days).
    :returns: An array of minimum values of the measurement for each day.
    """
    return np.min(data, axis=0)


def compute_standard_deviation_by_day(data):
    '''Computes standard deviation of average means for each day.'''
    means_by_day = map(daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))
    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)
    return daily_standard_deviation


def analyse_data(data_source):
    """Calculates the standard deviation by day between datasets.
    Gets all the inflammation data from CSV files within a directory, works out the mean
    inflammation value for each day across all datasets, then visualises the
    standard deviation of these means on a graph."""

    data = data_source.load_inflammation_data()

    daily_standard_deviation = compute_standard_deviation_by_day(data)

    graph_data = {
        'standard deviation by day': daily_standard_deviation,
    }
    return graph_data
    #views.visualize(graph_data)