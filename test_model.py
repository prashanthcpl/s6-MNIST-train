import torch
import sys
from model import Net

def test_parameter_count():
    """Test if model has less than 20k parameters"""
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f'Model has {total_params} parameters, should be less than 20000'
    print(f'Parameter count check passed. Total parameters: {total_params}')

def test_batch_normalization():
    """Test if model uses batch normalization"""
    model = Net()
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, 'Model should include BatchNormalization'
    print('Batch normalization check passed')

def test_dropout():
    """Test if model uses dropout"""
    model = Net()
    has_dropout = any(isinstance(m, torch.nn.Dropout2d) for m in model.modules())
    assert has_dropout, 'Model should include Dropout'
    print('Dropout check passed')

def test_gap_no_fc():
    """Test if model uses GAP and no FC layers"""
    model = Net()
    has_gap = any(isinstance(m, torch.nn.AvgPool2d) for m in model.modules())
    has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert has_gap, 'Model should use Global Average Pooling'
    assert not has_fc, 'Model should not use Fully Connected layers'
    print('Architecture check passed (GAP used, no FC layers)')

if __name__ == '__main__':
    test_parameter_count()
    test_batch_normalization()
    test_dropout()
    test_gap_no_fc()
    print('\n✓ All architecture checks passed') 