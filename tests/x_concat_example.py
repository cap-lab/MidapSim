from models.model_builder import ModelBuilder
from config import cfg
from test_wrapper import TestWrapper

def concat_test_example():
    mb = ModelBuilder("x_concat_test")
    x = mb.set_input_tensor(tensor_shape = (1, 128, 8, 8))
    x = mb.Conv(x, 128, 128, 1, 1, 0)
    x1 = mb.Conv(x, 128, 128, 1, 1, 0)
    x2 = mb.Conv(x, 128, 64, 1, 1, 0)
    concat = mb.Concat([x1, x2], axis='C')
    x1 = mb.Conv(concat, 192, 64, 1, 1, 0)
    x2 = mb.Conv(concat, 192, 80, 1, 1, 0)
    concat = mb.Concat([x1, x2], axis='Z')
    x = mb.Conv(concat, 144, 64, 1, 1, 0)
    return mb


def x_concat_test_example():
    mb = ModelBuilder("x_concat_test")
    x = mb.set_input_tensor(tensor_shape = (1, 128, 8, 8))
    x = mb.Conv(x, 128, 128, 1, 1, 0)
    x1 = mb.Conv(x, 128, 128, 1, 1, 0)
    x2 = mb.Conv(x, 128, 128, 1, 1, 0)
    concat = mb.Concat([x1, x2], axis='X')
    x1 = mb.Crop(concat, [1, -1])
    x1 = mb.Conv(x1, 128, 64, 1, 1, 0)
    x2 = mb.Conv(concat, 128, 64, 1, 1, 0)
    concat = mb.Concat([x1, x2], axis='W')
    x = mb.Conv(concat, 64, 32, 1, 1, 0)
    return mb
    
if __name__ == '__main__':
    tr = TestWrapper()
    print("-------------------------------------Test 1------------------------------------")
    test1 = concat_test_example()
    tr.run_all(test1)
    print("-------------------------------------Test 2------------------------------------")
    test2 = x_concat_test_example()
    tr.run_all(test2)
    print("-------------------------------Crop Test finished------------------------------")
