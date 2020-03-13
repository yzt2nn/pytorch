import os
import tempfile
from string import Template
import copy
import unittest
import warnings
import inspect
import re

import torch
from torch._six import PY2
import torch.testing._internal.common_utils as common
import torch.testing._internal.common_nn as common_nn
from torch.testing._internal.common_cuda import TEST_CUDA
import torch.utils.cpp_extension
from cpp_api_parity.parity_table_parser import parse_parity_tracker_table
from cpp_api_parity import sample_module, torch_nn_modules
from cpp_api_parity import functional_args_check
from cpp_api_parity import functional_impl_check
from cpp_api_parity import module_ctor_args_check
from cpp_api_parity import module_impl_check

class TestCppApiParity(common.TestCase):
  pass

parity_table_path = os.path.join(os.path.dirname(__file__), 'cpp_api_parity/parity-tracker.md')

parity_table = parse_parity_tracker_table(parity_table_path)

# yf225 TODO: use these lists instead!
# module_tests = sample_module.module_tests + common_nn.module_tests + common_nn.new_module_tests
# criterion_tests = common_nn.criterion_tests + common_nn.new_criterion_tests

module_tests = [common_nn.module_tests[0]] # yf225 TODO: change back to actual lists
criterion_tests = [] # yf225 TODO: change back to actual lists

module_impl_check.add_tests(TestCppApiParity, module_tests, criterion_tests, torch_nn_modules, parity_table)
# module_ctor_args_check.add_tests(module_tests, criterion_tests, torch_nn_modules, parity_table)
# functional_impl_check.add_tests(module_tests, criterion_tests, torch_nn_modules, parity_table)
# functional_args_check.add_tests(module_tests, criterion_tests, torch_nn_modules, parity_table)

if __name__ == "__main__":
  common.run_tests()
