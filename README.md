# Realsense-K4A_depth_comparison

# Overview
All the required codes for data capture to the analysis of all the experiments mentioned in the research paper **Comparing Depth Estimation of Azure Kinect and Realsense D435i Cameras** are present in the `codes/` directory.

# File Structure

```bash
└── code
    ├── config
    │   ├── azurekinect.json
    │   ├── default_config.json
    |   └── realsense.json
    ├── experiments
    │   ├── ambiance.py
    │   ├── camera_distance.py
    │   ├── object_color.py
    │   ├── object_motion.py
    │   ├── object_texture.py
    │   └── temperature.py
    ├── libs
    │   ├── experiments.py
    │   ├── type_convert.py
    │   └── visualization.py
    ├── recorder
    │   ├── azure_kinect_mkv_reader.py
    │   ├── azure_kinect_recorder.py
    │   ├── realsense_recorder.py
    │   └── viewer_realsense.py
    └── utils
    |    ├── file.py
    |    └── initialize_config.py
    └── requirements.txt
```
# Data Capture
The `recorder/` directory contains all the required Python codes for data capture for both cameras.

## Azure Kinect
**Record Data:**
```
python3 recorder/azure_kinect_recorder.py --config config/azurekinect.json --output <output.mkv> -a --len <time_in_sec>
```
**Export Data:**
```
python3 recorder/azure_kinect_mkv_reader.py --input <path-to-captured-mkv-file> --output <path-to-output-dir>
```

## RealSense
```
python3 recorder/realsense_recorder.py --config config/realsense.json --record_imgs --iteration <nth-iteration> --len <time_in_sec> --npy <path-to-save-data> --start <x1 y1 :starting-pixel-of-frame> --end <x2 y2 :ending-pixel-of-frame>
``` 

# Experiments
`libs/` and `experiments/` directories contain all the required Python codes for data visualization and analysis for each experiment.

Azure Kinect outputs are `.mkv` video, and `color` and `depth` images, so we need to extract the depth data from each depth image in the `depth/` directory. For that, we have added an `export_depth` flag in each experiment code which will render a color image. When one/multiple rectangular ROIs are drawn on that color image, depth data within the ROIs will be extracted for all the depth images within the `depth/` directory of the corresponding `color/` directory and saved as a `.npy` file. 


### Export Azure Kinect depth
```
python3 experiments/<experiment_name>.py --input <path-to-color-dir> <path-to-depth-dir> --npy <npy-filename> --export_depth --camera_id 1
```
### Visualize data
```
python3 experiments/<experiment_name>.py --npy <path-to-single/multiple-npy-files> --camera_id 0 (realsense) or 1 (kinect) --fig_path <path-to-save-image>
```
The visualizations are different for different experiments, so you will need to see the code in detail for that. However, in general, we have 3 different visualizations:

1. Heatmap
2. Temporal Plot
3. Distribution Check

For each of these, we have respective flags. You will need to set them to `True` as per your need.

# Contact and Citations
For pre-recorded data and further information, please contact the authors through the paper.
The paper is published on LNNS lecture series of the Springer Nature as a proceeding of Ninth International Congress on Information and Communication Technology (ICICT) 2024.
```
@inproceedings{rijal2024comparing,
  title={Comparing Depth Estimation of Azure},
  author={Rijal, Sanjay and Pokhrel, Suruchi and Om, Madhav and Ojha, Vaghawan},
  booktitle={Proceedings of Ninth International Congress on Information and Communication Technology: ICICT 2024, London, Volume 10},
  pages={225},
  organization={Springer Nature}
}
```
The full version pre-print of the paper is available at [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4597442).
