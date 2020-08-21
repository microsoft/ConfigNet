# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""AzureML-related utility scripts"""

from azureml.core.run import Run, _OfflineRun
import sys

def get_aml_run():
    run = Run.get_context()
    # If running locally or on Philly we will use an alternative logging method
    if type(run) == _OfflineRun:
        return None
    else:
        return run

def log_job_params(aml_run, args):
    if aml_run is None:
        return

    arg_dict = vars(args)
    aml_run.add_properties(arg_dict)

def log_losses(aml_run, names, values, prefix):
    assert len(names) == len(values)

    for name, value in zip(names, values):
        aml_run.log(prefix + name, float(value))