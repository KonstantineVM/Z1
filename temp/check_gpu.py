#!/usr/bin/env python
"""
Comprehensive GPU/CUDA availability check for PyMC and PyTensor
Tests GPU setup and provides diagnostic information
"""

import sys
import os
import subprocess
import numpy as np

print("=" * 60)
print("GPU/CUDA Availability Check for PyMC & PyTensor")
print("=" * 60)

# 1. Check Python and system info
print("\n1. SYSTEM INFORMATION:")
print("-" * 40)
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")

# 2. Check NVIDIA driver and CUDA
print("\n2. NVIDIA/CUDA SYSTEM CHECK:")
print("-" * 40)

# Check nvidia-smi
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ NVIDIA driver is installed")
        # Extract driver version
        for line in result.stdout.split('\n'):
            if 'Driver Version' in line:
                print(f"  {line.strip()}")
                break
    else:
        print("✗ nvidia-smi not found - NVIDIA driver may not be installed")
except FileNotFoundError:
    print("✗ nvidia-smi command not found")

# Check CUDA version
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ CUDA compiler (nvcc) is installed")
        # Extract CUDA version
        for line in result.stdout.split('\n'):
            if 'release' in line:
                print(f"  {line.strip()}")
except FileNotFoundError:
    print("✗ nvcc not found - CUDA toolkit may not be installed")

# 3. Check PyTensor installation and configuration
print("\n3. PYTENSOR CHECK:")
print("-" * 40)

try:
    import pytensor
    print(f"✓ PyTensor version: {pytensor.__version__}")
    
    # Check PyTensor configuration
    print(f"  Default device: {pytensor.config.device}")
    print(f"  Float type: {pytensor.config.floatX}")
    print(f"  Optimizer: {pytensor.config.optimizer}")
    
    # Check available devices
    from pytensor.gpuarray import pygpu
    print("\n  Checking PyGPU backend...")
    
    try:
        import pygpu
        print(f"  ✓ PyGPU version: {pygpu.__version__}")
        
        # List available devices
        ctx = pygpu.init('cuda')
        print(f"  ✓ CUDA device initialized: {ctx}")
    except Exception as e:
        print(f"  ✗ PyGPU initialization failed: {e}")
        
except ImportError as e:
    print(f"✗ PyTensor not installed or import error: {e}")

# 4. Test PyTensor GPU computation
print("\n4. PYTENSOR GPU COMPUTATION TEST:")
print("-" * 40)

try:
    import pytensor
    import pytensor.tensor as pt
    
    # Set device to GPU
    os.environ['PYTENSOR_FLAGS'] = 'device=cuda,floatX=float32'
    
    # Reimport to apply settings
    import importlib
    importlib.reload(pytensor.config)
    
    print(f"Testing with device={pytensor.config.device}, floatX={pytensor.config.floatX}")
    
    # Create a simple computation
    x = pt.matrix('x')
    y = pt.matrix('y')
    z = pt.dot(x, y)
    
    # Compile function
    f = pytensor.function([x, y], z)
    
    # Test data
    x_val = np.random.randn(1000, 1000).astype(np.float32)
    y_val = np.random.randn(1000, 1000).astype(np.float32)
    
    # Run computation
    import time
    start = time.time()
    result = f(x_val, y_val)
    gpu_time = time.time() - start
    
    print(f"✓ GPU computation successful!")
    print(f"  Matrix multiplication (1000x1000): {gpu_time:.4f} seconds")
    
    # Compare with CPU
    os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float32'
    importlib.reload(pytensor.config)
    
    f_cpu = pytensor.function([x, y], z)
    start = time.time()
    result_cpu = f_cpu(x_val, y_val)
    cpu_time = time.time() - start
    
    print(f"  CPU time: {cpu_time:.4f} seconds")
    print(f"  GPU speedup: {cpu_time/gpu_time:.2f}x")
    
except Exception as e:
    print(f"✗ GPU computation test failed: {e}")

# 5. Check PyMC
print("\n5. PYMC CHECK:")
print("-" * 40)

try:
    import pymc as pm
    print(f"✓ PyMC version: {pm.__version__}")
    
    # Test PyMC with GPU
    print("\n  Testing PyMC model on GPU...")
    
    # Reset to GPU
    os.environ['PYTENSOR_FLAGS'] = 'device=cuda,floatX=float64'
    import importlib
    import pytensor
    importlib.reload(pytensor.config)
    
    # Simple model
    with pm.Model() as model:
        x = pm.Normal('x', mu=0, sigma=1, shape=100)
        y = pm.Normal('y', mu=x, sigma=1, observed=np.random.randn(100))
        
        # Try to sample (just a few draws to test)
        trace = pm.sample(
            draws=100,
            tune=50,
            chains=1,
            cores=1,
            progressbar=False,
            return_inferencedata=False
        )
    
    print("  ✓ PyMC sampling with GPU successful!")
    print(f"  Samples shape: {trace['x'].shape}")
    
except ImportError:
    print("✗ PyMC not installed")
except Exception as e:
    print(f"✗ PyMC GPU test failed: {e}")

# 6. Check other GPU libraries
print("\n6. OTHER GPU LIBRARIES:")
print("-" * 40)

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("- PyTorch not installed")

# Check JAX
try:
    import jax
    print(f"✓ JAX version: {jax.__version__}")
    print(f"  GPU devices: {jax.devices('gpu')}")
except ImportError:
    print("- JAX not installed")

# 7. Recommendations
print("\n7. RECOMMENDATIONS:")
print("-" * 40)

# Check if GPU is properly set up for PyTensor
gpu_available = False
try:
    import pytensor
    os.environ['PYTENSOR_FLAGS'] = 'device=cuda,floatX=float64'
    importlib.reload(pytensor.config)
    if pytensor.config.device == 'cuda':
        gpu_available = True
except:
    pass

if gpu_available:
    print("✓ GPU is properly configured for PyTensor/PyMC!")
    print("\nTo use GPU in your script, add at the top:")
    print("  import os")
    print("  os.environ['PYTENSOR_FLAGS'] = 'device=cuda,floatX=float64'")
else:
    print("✗ GPU is not properly configured for PyTensor/PyMC")
    print("\nPossible solutions:")
    print("1. Install CUDA toolkit (https://developer.nvidia.com/cuda-toolkit)")
    print("2. Install cuDNN (https://developer.nvidia.com/cudnn)")
    print("3. Install PyTensor with GPU support:")
    print("   conda install -c conda-forge pytensor pygpu")
    print("4. Set CUDA_HOME environment variable")
    print("5. Ensure CUDA/bin is in your PATH")

# 8. Environment variables
print("\n8. RELEVANT ENVIRONMENT VARIABLES:")
print("-" * 40)
env_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH', 'PATH', 'PYTENSOR_FLAGS']
for var in env_vars:
    value = os.environ.get(var, 'Not set')
    if var == 'PATH' and value != 'Not set':
        # Only show CUDA-related paths
        cuda_paths = [p for p in value.split(':') if 'cuda' in p.lower()]
        value = ':'.join(cuda_paths) if cuda_paths else 'No CUDA paths found'
    print(f"{var}: {value}")

print("\n" + "=" * 60)
print("Check complete!")
print("=" * 60)

# Create a summary
print("\nSUMMARY:")
if gpu_available:
    print("✓ Your system is ready for GPU-accelerated PyMC/PyTensor!")
else:
    print("✗ GPU acceleration is not available. Running on CPU.")
