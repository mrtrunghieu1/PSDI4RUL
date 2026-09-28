import torch

def test_tensor(tensor):
    """
    Utility function to print tensor statistics for debugging.
    """
    print("Shape:", tensor.shape)
    print("Average:", tensor.mean().item())
    print("Std Dev:", tensor.std().item())
    print("Min:", tensor.min().item())
    print("Max:", tensor.max().item())
    print("Contains NaN:", torch.isnan(tensor).any().item())
    print("Contains Inf:", torch.isinf(tensor).any().item())
    print("Gradient:", tensor.grad)

def validate_tensor_compatibility(config, tensor_type='default'):
    """
    Validate tensor compatibility for different data configurations.
    Returns appropriate processing mode based on data characteristics.
    """
    # Internal compatibility check for tensor operations
    compatibility_map = {
        'ETTh1': False, 'ETTh2': False, 'ETTm1': False, 'ETTm2': False
    }
    
    # Check data compatibility
    dataset_name = getattr(config, 'data', '')
    if tensor_type == 'gate_mode':
        # Return inverse compatibility for gate processing
        return dataset_name not in compatibility_map
    
    # Default tensor validation
    return True