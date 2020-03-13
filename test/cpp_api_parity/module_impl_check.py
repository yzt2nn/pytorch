import torch
import torch.testing._internal.common_nn as common_nn
from cpp_api_parity.utils import TorchNNTestParams

# Check 1: Module implementation correctness check:

# Step 1: Translate ctor args from Python layer to C++ layer
# Step 2: Construct a C++ layer, run forward and backward on it, save all its params/buffers/gradients into a ScriptModule
# Step 3: Load that ScriptModule into Python, and compare output/params/buffers/gradients with Python layer (forward and backward)

# yf225 TODO: move to common utils?
devices = ['cpu', 'cuda']

# yf225 TODO: move to common utils?
def _compute_module_name(test_params_dict):
    fullname = test_params_dict.get('fullname', None)
    if fullname:
        # NOTE: This doesn't work for some of the `wrap_functional` module tests such as "interpolate_nearest_1d",
        # because in that case the module `interpolate` is not in `torch.nn` but rather in `torch.nn.functional`.
        # We will fix this when we have parity tests for `torch.nn.functional` modules.
        module_name = fullname.split('_')[0]
    else:
        module_name = test_params_dict.get('module_name')
    return module_name

# yf225 TODO: move to common utils?
def _process_test_params_for_module(test_params_dict, module_metadata, device, is_criterion):
    module_name = _compute_module_name(test_params_dict)
    test_params_dict['constructor'] = test_params_dict.get('constructor', getattr(torch.nn, module_name))
    if is_criterion:
        test = common_nn.CriterionTest(**test_params_dict)
    else:
        test = common_nn.ModuleTest(**test_params_dict)
    # yf225 TODO: can we remove the magic number `5` here?
    module_variant_name = test.get_name()[5:] + (('_' + device) if device != 'cpu' else '')

    return TorchNNTestParams(
        module_name=module_name,
        module_variant_name=module_variant_name,
        test_instance=test,
        cpp_constructor_args=test_params_dict.get('cpp_constructor_args'),
        has_parity=test_params_dict.get('has_parity', True),
        device=device,
    )

# yf225 TODO: move to common utils?
def has_test(unit_test_class, test_name):
    return hasattr(unit_test_class, test_name)

# yf225 TODO: move to common utils?
def add_test(unit_test_class, test_name, test_fn):
    if has_test(test_name):
        raise RuntimeError("Found two tests with the same name: " + test_name)
    setattr(unit_test_class, test_name, test_fn)

def add_torch_nn_module_impl_parity_tests(parity_table, unit_test_class, torch_nn_modules, module_tests, is_criterion):
  torch_nn_test_params_map = {}
  for test_params_dict in module_tests:
    # Skip all `torch.nn.functional` tests, since they are handled by another test suite.
    if 'FunctionalModule' in str(test_params_dict.get('constructor', '')):
      continue

    module_name = _compute_module_name(test_params_dict)

    assert hasattr(torch.nn, module_name), \
      "`torch.nn` doesn't have module `{}`. ".format(module_name) + \
      "If you are adding a new test, please set `fullname` using format `ModuleName_desc`, " + \
      "or set `module_name` using format `ModuleName`."

    module_full_name = 'torch::nn::' + module_name
    # If equivalent module in C++ frontend doesn't exist, we don't do the parity test.
    if module_full_name not in parity_table['torch::nn']:
      continue

    has_impl_parity, _ = parity_table['torch::nn'][module_full_name]

    def add_variant_test_for_module(module_name, test_params_dict, has_impl_parity, torch_nn_modules):
      module_metadata = torch_nn_modules.module_metadata_map[module_name]
      for device in devices:
        test_params = _process_test_params_for_module(
          test_params_dict=test_params_dict,
          module_metadata=module_metadata,
          device=device,
          is_criterion=is_criterion)
        test_name = 'test_torch_nn_{}'.format(test_params.module_variant_name)
        torch_nn_test_params_map[test_name] = test_params

        print(test_params) # yf225 TODO: debug

        # def test_fn(self):
        #   self._test_torch_nn_module_variant(test_params=torch_nn_test_params_map[self._testMethodName])

        # if device == 'cuda':
        #   test_fn = unittest.skipIf(not TEST_CUDA, "CUDA unavailable")(test_fn)

        # # If `Implementation Parity` entry in parity table for this module is `No`,
        # # we mark the test as expected failure.
        # if not has_impl_parity:
        #   test_fn = unittest.expectedFailure(test_fn)

        # add_test(unit_test_class, test_name, test_fn)

    add_variant_test_for_module(
      module_name=module_name,
      test_params_dict=test_params_dict,
      has_impl_parity=has_impl_parity,
      torch_nn_modules=torch_nn_modules)

def add_tests(unit_test_class, module_tests, criterion_tests, torch_nn_modules, parity_table):
  add_torch_nn_module_impl_parity_tests(
    parity_table=parity_table,
    unit_test_class=unit_test_class,
    torch_nn_modules=torch_nn_modules,
    module_tests=module_tests,
    is_criterion=False)

  add_torch_nn_module_impl_parity_tests(
    parity_table=parity_table,
    unit_test_class=unit_test_class,
    torch_nn_modules=torch_nn_modules,
    module_tests=criterion_tests,
    is_criterion=True)
