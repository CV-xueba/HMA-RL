import importlib


def build_operation(opt):
    operation = opt["env"]["operation"]["name"]
    operation_module = importlib.import_module('.', package='HMARL.env.operations.{}'.format(operation))
    Operation = getattr(operation_module, "AdaptiveBilateralFiltering")
    operation = Operation(opt)
    return operation
