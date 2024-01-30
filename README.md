# Open-RadVLAD: Fast and Robust Radar Place Recognition

This is an open implementation and benchmark from the paper at 

**Open-RadVLAD: Fast and Robust Radar Place Recognition**
M. Gadd and P. Newman <br>
*IEEE Radar Conference (RadarConf), 2024* <br>
[https://arxiv.org/abs/2401.15380](https://arxiv.org/abs/2401.15380)

If you use this code in your research, please cite the following paper:

```bash
@inproceedings{gadd2024openradvlad,
title={{Open-RadVLAD: Fast Shift and Rotation Invariant Radar Place Recognition}},
author={Gadd, Matthew and Newman, Paul},
booktitle={IEEE Radar Conference (RadarConf)},
year={2024}
}
```

An example use is shown below, for the FFT-RadVLAD method

```bash
python -m launch.fft_vlad
```

This runs over all pairs of trajectories from [The Oxford Radar RobotCar Dataset](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/).