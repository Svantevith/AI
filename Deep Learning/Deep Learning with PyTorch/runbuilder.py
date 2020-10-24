from itertools import product
from collections import OrderedDict, namedtuple
from typing import OrderedDict


class RunBuilder:
    @staticmethod
    def get_runs(param_dict: OrderedDict):
        Run = namedtuple('Run', param_dict.keys())  # 'Run' - typename, dict.keys() - field names
        runs = []
        for values in product(*param_dict.values()):  # *args, because the values in dict are lists of different sizes
            runs.append(
                Run(*values)  # we pass *args to the named tuple, so each field can be connected with the final product
            )

        return runs
