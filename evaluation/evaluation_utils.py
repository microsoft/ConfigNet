# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Scripts containing common code for ConfigNet evaluation and visualization scripts"""
import numpy as np
import os
from pathlib import Path
import re

def dnn_filename_prompt():
    # Load tkinter here as it may not load correctly on Linux
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=(("json files", "*.json"),))
    root.destroy()

    return file_path

def directory_prompt():
    # Load tkinter here as it may not load correctly on Linux
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory()
    root.destroy()

    return dir_path

def get_model_paths(model_path_or_dir, names_with_digits_only=True):
    if os.path.isfile(model_path_or_dir):
        return [model_path_or_dir]

    model_paths = list(Path(model_path_or_dir).glob("**/*.json"))
    model_paths = [str(path) for path in model_paths]

    if names_with_digits_only:
        filtered_paths = []
        for path in model_paths:
            if re.match(".*[0-9]+.json", path):
                filtered_paths.append(path)
        model_paths = filtered_paths

    return model_paths