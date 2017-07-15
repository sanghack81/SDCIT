We describe how to reproduce results in (Lee and Honavar, 2017).
We will assume that `kcipt` and `sdcit` are installed under `~/Downloads`.

PREPARATION
=====

### Setting up MATLAB. 

Reproducing experiments requires MATLAB since `CHSIC`, `KCIT`, and `KCIPT` are all implemented with MATLAB.

Note that reproducing all experiments may require about **a week or two** depending on your computing power.
All experiments were done in macOS Sierra and Ubuntu 16.04.

  
###Setting up `KCIPT`

[`KCIPT`](https://github.com/garydoranjr/kcipt) is available in github. After cloning the repository, follow the setup script provided in the repository.
The shell script below will download `KCIPT` and other necessary files (e.g., `KCIT` and `CHSIC`) in `~/Downloads/kcipt` directory.
  
  
(If you are using macOS and does not have `wget`, you may want to install `wget` from homebrew.) 
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install wget
```
 
```
cd ~/Downloads
git clone https://github.com/garydoranjr/kcipt
./setup.sh
```

### Configure `exp_setup.py`

Information about where data, results, and figures will be stored needs to be configured. Edit following variables in `experiments/exp_setup.py` of `SDCIT`:
 - `SDCIT_WORKING_DIR`: `experiments` directory where `SDCIT` is installed.
 - `SDCIT_DATA_DIR`: where data will be stored. Since resulting data can be around 7GB, you may need to avoid, e.g., directory under dropbox if you do not want to sync large number of files. You will need to change some other files (which will be described below.)


### Copy matlab scripts in `SDCIT` to `KCIPT`

We wrote scripts: to store data used in `KCIPT`; and to run `KCIT` with different parameters. All files are under `matlab` directory in `SDCIT` package. 

```
cp -R ~/Downloads/SDCIT/matlab/* ~/Downloads/kcipt/
```


### Generating Data

In `~/Downloads/kcipt` directory, there are three matlab files in the form of `save_*.m`, which generate necessary data to be used in the python implementation of `SDCIT` and `KCIPT`.

If you are using recent version of MATLAB, you are most likely to change `setDefaultStream` to `setGlobalStream` in `~/Downloads/kcipt/data/synthetic.m`

By default, current files will generate data under `~/kcipt_data` directory. If you want to change the directory, please modify three matlab files starting with `save_` accordingly. Please check `SDCIT_DATA_DIR` variable set correctly.
Note again that the resulting data will be around **7 GB**.

- `save_chaotic.m` generates a set of chaotic time series datasets.
- `save_postnonlinear.m` generates a set of post-nonlinear noise datasets.
- `save_postnonlinear_distance.m` generates a set of distance matrix to be used in `SDCIT`.


### Clean `results` and `figures`

All results and figures used in the paper are placed in `~/Downloads/SDCIT/experiments/results` and `~/Downloads/SDCIT/experiments/figures` directories. If you decided to replicate our experiments, you may delete two directories.
 
 
RUNNING EXPERIMENTS
=====


### Run CHSIC, KCIT (original), and KCIT (fixed)

The results of three algorithm can be obtained by running matlab scripts provided in `KCIPT`. Find `run_xxxxx_chaotic.m` and `run_xxxxx_postnonlinear.m` in the base directory of `kcipt`. These will take about a few days to finish. You may open several MATLAB instances and run each script separately. 
 
 Depending on the version of MATLAB, you may need to change random number generator. Further, the simplex algorithm used in `KCIPT`  may not be available. In fact, (as described in our paper), we used our own implementation of `KCIPT`.

In `~/Downloads/kcipt/results` directory, you will find `*.csv` files for `KCIT` (original)  and `CHSIC`. Copy these files to `~/Downloads/SDCIT/experiments/results` directory to be analyzed later.

- `run_chsic_chaotic.m` runs `CHSIC` on chaotic time series data
- `run_chsic_postnonlinear.m` runs `CHSIC` on postnonlinear-noise data
- `run_kcit_chaotic.m` runs `KCIT` on chaotic time series data
- `run_kcit_postnonlinear.m` runs `KCIT` on postnonlinear-noise data
- `run_kcit2_chaotic.m`  runs fixed `KCIT` on chaotic time series data
- `run_kcit2_postnonlinear.m` runs fixed `KCIT` on postnonlinear-noise data
- `run_kcit_chaotic_timing.m` estimates run time for `KCIT`

You do not need to run `run_kcipt_*.m` since 
- We provide **faster** python implementation of `KCIPT`
- A simplex algorithm is [**removed**](https://www.mathworks.com/help/optim/release-notes.html?rntext=&startrelease=R2016b&endrelease=R2016b&groupby=release&sortby=descending&searchHighlight=) from MATLAB since 2016.

### Run KCIPT and SDCIT

Python scripts are in `experiments` of `SDCIT`:

- `run_SDCIT.py` runs `SDCIT` on both datasets.
- `run_KCIPT.py` runs `KCIPT` on both datasets.
- `run_SDCIT_wo_adjust.py` runs `SDCIT` without error-adjustment on both datasets. (to replicate results in supplementary material.)
- `run_c_SDCIT.py` runs the C implementation of `SDCIT` on both datasets.  
- `run_compare_sdcit_kcipt_power.py` runs tests required to (Figure 1, 5, and 6) and Section 5.4). The script needs results from `run_KCIPT.py`. 
- `run_time_sdcit.py` runs tests for runtime estimation. (Table 1.)
- `run_kernel_choice_sensitivity.py` runs tests for Figure 4. 
- `run_viz_adjust.py` runs tests to compare the effect of permutation error-based adjustment.


Running all experiments will likely take a few days. 

### Generate figures from results

Run `draw_figures.py` to generate Figure 2 and 3.
 
 
 Note that scripts not described in this page are likely adjunct tests. You can ignore them.