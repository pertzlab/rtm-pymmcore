# rtm-pymmcore  
Real-time feedback control microscopy using `pymmcore` as an interface.  

## Overview  
This repository enables real-time cell segmentation and feature extraction during microscopy experiments. Instead of relying on post-processing, this approach provides immediate feedback control. For example, if spatial stimulation is available, stimulation areas can be dynamically defined based on cell morphology.  

Another key application is real-time segmentation and feature extraction to streamline experiments and minimize post-processing requirements.  

## Information  
This repository is a work in progress and leverages `pymmcore-plus` to control the microscope and acquire images. The workflow follows these main steps:  

1. **Defining Acquisition Positions & Stimulation Parameters**  
   - A dataframe is generated to store acquisition positions, which can be selected using `pymmcore-plus` GUI widgets.  
   - Stimulation parameters (e.g., pulse duration, intensity) are defined and configured within the Jupyter notebook files.  

2. **Image Acquisition & Processing**  
   - The dataframe is passed to the core of `rtm-pymmcore`, which handles image acquisition, segmentation, and feature extraction.  
   - Optionally, an external segmentation engine (running on another computer or instance via `imaging-server-kit`) can be used.  
   - Single-cell results are stored in a dataframe.  

3. **Data Visualization & Analysis**  
   - The `viewer` script allows visualization of individual fields of view (FOVs) along with their corresponding stimulation conditions.  
   - The `data_analysis_plotting` scripts facilitate report generation.  

### Microscope Setup  
The workflow relies on a **uManager configuration file** for microscope setup. The configuration file must include:  

- A `setup` group with a `Startup` preset (executed at script initialization).  
- Additional presets for each fluorophore used in the experiment (e.g., filter or laser selection).  

### Running the Script  
- The workflow is executed step by step via Jupyter notebooks.  
- Currently, only **full field of view (FOV) stimulation** is implemented, but the modular code structure allows easy adaptation to other stimulation patterns.  
- Two versions of the `01_ERK-KTR_full_fov_stimulation` scripts exist for different microscope setups:  
  - `01_ERK-KTR_full_fov_stimulation_Jungfrau` (recommended for a general overview).  
  - `01_ERK-KTR_full_fov_stimulation_Niesen`, which includes an additional routine to prevent laser sleep.  
