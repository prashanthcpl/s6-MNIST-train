import torch
import sys
import inspect
from model import Net

def check_parameter_count():
    """Check if model architecture has less than 20k parameters without instantiating"""
    model_class = Net
    # Create a dummy instance just to count parameters
    model = model_class()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f'Model has {total_params} parameters, should be less than 20000'
    print(f'Parameter count check passed. Total parameters: {total_params}')

def check_batch_normalization():
    """Check if BatchNorm is in model definition"""
    model_class = Net
    model_code = inspect.getsource(model_class)
    assert 'BatchNorm2d' in model_code, 'Model should include BatchNormalization'
    print('Batch normalization check passed')

def check_dropout():
    """Check if Dropout is in model definition"""
    model_class = Net
    model_code = inspect.getsource(model_class)
    assert 'Dropout' in model_code, 'Model should include Dropout'
    print('Dropout check passed')

def check_gap_no_fc():
    """Check if model uses GAP and no FC layers"""
    model_class = Net
    model_code = inspect.getsource(model_class)
    assert 'AvgPool2d' in model_code, 'Model should use Global Average Pooling'
    assert 'Linear' not in model_code, 'Model should not use Fully Connected layers'
    print('Architecture check passed (GAP used, no FC layers)')

if __name__ == '__main__':
    check_parameter_count()
    check_batch_normalization()
    check_dropout()
    check_gap_no_fc()
    print('\n✓ All architecture checks passed') 