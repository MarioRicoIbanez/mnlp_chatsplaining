#!/usr/bin/env python3

import torch
import subprocess
import time

def get_gpu_memory():
    """Get GPU memory usage in GB"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        free = total - allocated
        return {
            'total': total,
            'allocated': allocated, 
            'cached': cached,
            'free': free
        }
    return None

def monitor_gpu_memory():
    """Monitor GPU memory continuously"""
    print("GPU Memory Monitor - Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        while True:
            memory = get_gpu_memory()
            if memory:
                print(f"Total: {memory['total']:.2f}GB | "
                      f"Allocated: {memory['allocated']:.2f}GB | "
                      f"Cached: {memory['cached']:.2f}GB | "
                      f"Free: {memory['free']:.2f}GB")
            else:
                print("CUDA not available")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    # Check current memory status
    print("Current GPU Memory Status:")
    memory = get_gpu_memory()
    if memory:
        print(f"Total: {memory['total']:.2f}GB")
        print(f"Allocated: {memory['allocated']:.2f}GB") 
        print(f"Cached: {memory['cached']:.2f}GB")
        print(f"Free: {memory['free']:.2f}GB")
        print()
        
        # Start monitoring
        monitor_gpu_memory()
    else:
        print("CUDA not available on this system") 