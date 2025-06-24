"""
save the config as a txt
"""
import os

def Saveconfig():
    dirScript = os.path.dirname(os.path.realpath(__file__))
    dirScript = dirScript.replace('/Utils', '')
    python_filename = dirScript + '/Configuration/' + 'Config.py'
    Txtdir = dirScript + '/output/'
    os.makedirs(Txtdir, exist_ok=True)
    
    text_filename = dirScript + '/output/' + 'Config.txt'

    # Read the contents of the Python file
    with open(python_filename, 'r') as py_file:
        content = py_file.read()

    # Write the contents to a text file
    with open(text_filename, 'w') as txt_file:
        txt_file.write(content)

