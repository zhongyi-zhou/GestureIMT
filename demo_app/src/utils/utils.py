import json
import os, sys
import time
import numpy as np
import math


"----------------------------- File I/O -----------------------------"
def readjson(filepath):
    with open(filepath, 'r') as json_file:
        mydict = json.load(json_file)
        print(f"load success...!!! at {filepath}")
    return mydict

def savejson(mydict, filepath):
    with open(filepath, 'w') as json_file:
        json.dump(mydict, json_file)
        print(f"save success...!!! at {filepath}")

def readtxt_all(filepath):
    with open(filepath, 'r') as txt_file:
        mylist = txt_file.readlines()
        print(f"load success...!!! at {filepath}")
    return mylist

def savetxt_all(mylist, filepath):
    with open(filepath, 'w') as txt_file:
        txt_file.writelines(mylist)
        print(f"save success...!!! at {filepath}")

"----------------------------- XXXX -----------------------------"
def path_separate(mypath):
    dirname = os.path.dirname(mypath)
    basename, ext = os.path.splitext(os.path.basename(mypath))
    return dirname, basename, ext


def get_name_by_time():
    timestr = time.strftime("%Y%m%d_%H%M%S")
    return timestr
    
def get_name_by_date():
    timestr = time.strftime("%Y%m%d")
    return timestr

def convert_n_digit(number, digit=4):
    myformat = '{:0>'+ str(digit) + '}'
    return myformat.format(str(number))


