import models.resnet as resnet
import models.mobilenet as mobilenet
from test_wrapper import TestWrapper
from models.model_builder import ModelBuilder

def get_origin_mb(tr):
    model = tr.midap_model
    builder = ModelBuilder()
    init_layer = model.init_layer
    for name in init_layer:
        data = model[name].main_op.output_tensor
        builder.set_input_tensor(name, input_tensor = data, order = 'WHC')
    x = init_layer[0]
    odict = tr.cv.operator_dict
    for op_name in odict:
        if op_name in init_layer:
            continue
        op = odict[op_name]
        builder.from_generic_op(op)
    return builder

if __name__ == '__main__':
    resnet50 = resnet.resnet50(input_size = 224)
    tr = TestWrapper()
    tr.setup_from_builder(resnet50)
    # tr.compile() Compile must not be done
    print('----------------------------- Setup finished ------------------------------------')
    builder = get_origin_mb(tr) 
    tr_mimic = TestWrapper()
    tr_mimic.run_all(builder)

    print('---------------------------- Resnet50 Test finished -----------------------------')
    mobilenet = mobilenet.mobilenet(input_size = 224)
    tr.setup_from_builder(mobilenet)
    
    print('----------------------------- Setup finished ------------------------------------')
    builder = get_origin_mb(tr) 
    tr_mimic = TestWrapper()
    tr_mimic.run_all(builder)
    
    print('---------------------------- Mobilenet Test finished -----------------------------')
