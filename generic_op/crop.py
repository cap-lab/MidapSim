from .operator_base import OperatorBase


class Crop(OperatorBase):
    def __init__(self, op_type='Crop', crop_x = None, crop_y = None, **kwargs):
        super(Crop, self).__init__(op_type=op_type, **kwargs)
        if crop_x is not None:
            if not isinstance(crop_x, list) and not isinstance(crop_x, tuple):
                raise ValueError("crop_x should be tuple or list")
            if len(crop_x) != 2:
                raise ValueError("len(crop_x) should be 2")
            if crop_x[1] > 0:
                raise ValueError("crop_x[1] should be zero or negative")
        if crop_y is not None:
            if not isinstance(crop_y, list) and not isinstance(crop_y, tuple):
                raise ValueError("crop_y should be tuple or list")
            if len(crop_y) != 2:
                raise ValueError("len(crop_y) should be 2")
            if crop_y[1] > 0:
                raise ValueError("crop_y[1] should be zero or negative")
        self.crop_x = crop_x
        self.crop_y = crop_y
