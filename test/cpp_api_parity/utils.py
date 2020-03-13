from collections import namedtuple

TorchNNTestParams = namedtuple(
    'TorchNNTestParams',
    [
        'module_name',
        'module_variant_name',
        'test_instance',
        'cpp_constructor_args',
        'has_parity',
        'device',
    ]
)

CppArg = namedtuple('CppArg', ['type', 'value'])
