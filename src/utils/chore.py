"""
Other utility functions.
"""
from datetime import datetime


def generate_now_datetime():
    return datetime.now().strftime("%Y%m%d-%H%M%S")
