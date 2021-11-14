# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 12:06:11 2021

@author: danie
"""

import pandas as pd
import os
import sys

dirname = os.path.abspath(os.path.dirname(sys.argv[0]))
filename = os.path.join(dirname, 'data/ds_challenge_2021.csv')

df = pd.read_csv(filename)

df.columns


