# HRIR Convolver on Bela

A real-time HRTF-based binaural audio simulator.
This project was made for the ECS7012P course at Queen Mary University of London.

https://github.com/yoyololicon/bela-hrir-convolver/assets/17811395/dbff7d2f-188f-4942-bb13-e8cc0a97cea8

## Quick Start

1. Grab a [Bela board](https://bela.io/).
2. Run the [preprocessing notebook](preprocess.ipynb). In the notebook, you can specify the HRTF `*.sofa` you want to download from [sofaconvention](https://www.sofaconventions.org/mediawiki/index.php/Files) and the directory name for putting the processed files.
3. Specify the directory name at L27 in `render.cpp` to the directory you just created.
4. Upload the whole working directory to your Bela board then you're done!

## Additional Materials

For details on the application design and ablation studies, please refer to this [report](report.pdf).
To use the minimum-phase and ITDs representation stated in the report, please replace `render.cpp` with the one at the branch `minimum-phase-itd`.