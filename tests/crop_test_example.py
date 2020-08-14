from models.model_builder import ModelBuilder
from config import cfg
from test_wrapper import TestWrapper

def crop_test_example1(y = False):
    mb = ModelBuilder("crop_test")
    x = mb.set_input_tensor(tensor_shape = (1, 128, 5, 5))
    x = mb.Conv(x, 128, 128, 1, 1, 0)
    x_crop = mb.Crop(x, [1,-1], None if not y else [1,-1])
    x = mb.Conv(x, 128, 64, 1, 1, 0)
    pad = [1, 0] if not y else [0,0]
    x = mb.Conv(x, 64, 64, 3, 1, pad)
    x = mb.Conv(x, 64, 128, 1, 1, 0, activation = 'Linear')
    x = mb.Sum(x, x_crop, activation = 'Relu')
    return mb

def crop_test_example2():
    mb = ModelBuilder("crop_test")
    x = mb.set_input_tensor(tensor_shape = (1, 128, 5, 5))
    x = mb.Conv(x, 128, 128, 1, 1, 0)
    x_crop = mb.Crop(x, [1,-1], [1,-1])
    x_crop = mb.Conv(x_crop, 128, 128, 1, 1, 0)
    x = mb.Conv(x, 128, 64, 1, 1, 0)
    x = mb.Conv(x, 64, 64, 3, 1, 0)
    x = mb.Conv(x, 64, 128, 1, 1, 0, activation = 'Linear')
    x = mb.Sum(x, x_crop, activation = 'Relu')
    return mb

def crop_test_example3():
    mb = ModelBuilder("crop_test")
    x = mb.set_input_tensor(tensor_shape = (1, 128, 5, 5))
    x = mb.Conv(x, 128, 128, 1, 1, 0)
    x_crop = mb.Crop(x, [1,-1])
    x_crop2 = mb.Crop(x, [1,-1])
    x = mb.Conv(x_crop2, 128, 64, 1, 1, 0)
    x = mb.Conv(x, 64, 64, 1, 1, 0)
    x = mb.Conv(x, 64, 128, 1, 1, 0, activation = 'Linear')
    x = mb.Sum(x, x_crop, activation = 'Relu')
    return mb

if __name__ == '__main__':
    tr = TestWrapper()
    print("-------------------------------------Test 1------------------------------------")
    mb_x_crop1 = crop_test_example1()
    tr.run_all(mb_x_crop1)
    print("-------------------------------------Test 2------------------------------------")
    mb_x_crop2= crop_test_example2()
    tr.run_all(mb_x_crop2)
    print("-------------------------------------Test 3------------------------------------")
    mb_x_crop3= crop_test_example3()
    tr.run_all(mb_x_crop3)
    print("-------------------------------Crop Test finished------------------------------")
