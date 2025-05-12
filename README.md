# GPU-Accelerated Recursive Bilateral Filtering

This project implements a GPU-accelerated version of Recursive Bilateral Filtering (RBF), originally developed by Qingxiong Yang. The implementation significantly improves performance by utilizing CUDA for parallel processing on NVIDIA GPUs.

## Overview

Recursive bilateral filtering is an edge-preserving filtering method that:
- Has linear computational complexity in both input size and dimensionality
- Preserves edges while smoothing the image
- Is significantly faster than traditional bilateral filtering methods

## Performance Results

We tested the implementation on various image sizes, comparing three different approaches:

1. CPU External Buffer
2. GPU Naive Kernel
3. GPU Refactored Kernel

### Performance Comparison (Average Times)

| Image Size | CPU External | GPU Naive | GPU Refactored | Speedup (vs CPU) |
|------------|--------------|-----------|----------------|------------------|
| 5760×3840  | 2.48s        | 1.50s     | 0.16s         | 15.5x           |
| 6016×4016  | 2.70s        | 1.62s     | 0.15s         | 18.0x           |
| 4288×2848  | 1.36s        | 0.83s     | 0.12s         | 11.3x           |
| 10800×5400 | 6.60s        | 3.66s     | 0.24s         | 27.5x           |
| 12039×4816 | 6.46s        | 3.62s     | 0.24s         | 26.9x           |
| 14805×4022 | 6.65s        | 3.77s     | 0.28s         | 23.8x           |
| 14524×7946 | 12.89s       | 7.80s     | 0.54s         | 23.9x           |
| 13347×7162 | 10.66s       | 6.04s     | 0.44s         | 24.2x           |
| 20919×6260 | 14.74s       | 8.26s     | 0.50s         | 29.5x           |
| 16376×5611 | 10.25s       | 5.99s     | 0.44s         | 23.3x           |

The GPU-accelerated implementation shows significant performance improvements:
- Average speedup of 22.4x compared to CPU implementation
- Refactored GPU kernel is approximately 5-10x faster than the naive GPU implementation
- Performance scales well with image size

## Implementation Details

The project includes three main implementations:

1. **CPU External Buffer**: Original CPU implementation with external buffer
2. **GPU Naive Kernel**: Initial GPU implementation with basic CUDA optimizations
3. **GPU Refactored Kernel**: Optimized GPU implementation with:
   - Efficient memory management
   - Optimized kernel design
   - Better thread utilization
   - Reduced memory transfers

## Requirements

- CUDA-capable NVIDIA GPU
- CUDA Toolkit
- C++ compiler with C++11 support

## Building and Running

```bash
cd example
make
./test.sh
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{yang2012recursive,
    title={Recursive bilateral filtering},
    author={Yang, Qingxiong},
    booktitle={European Conference on Computer Vision},
    pages={399--413},
    year={2012},
    organization={Springer}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
