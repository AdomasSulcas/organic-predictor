"""Suppress various warnings from dependencies. Edit this file to add or remove warnings as needed.
    Current iteration does not use this file, but it is kept for future use."""

import warnings
import logging
import os

# Suppress Prophet warnings
os.environ['PROPHET_SUPPRESS_PLOTLY_WARNING'] = '1'
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='Importing plotly failed. Interactive plots will not work.')
warnings.filterwarnings('ignore', category=FutureWarning)