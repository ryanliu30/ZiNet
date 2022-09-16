import numpy as np
import csv
import pandas as pd

pd.read_csv("./edu_standard.txt", dtype = object, sep = '\t').to_csv("./edu_standard.csv")