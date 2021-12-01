# `bfosc`
An automated pipeline for the 2.16m BFOSC data reduction.

## Installation

1. install `songcn` package
    - `pip install -U git+git://github.com/hypergravity/songcn`
    - `pip show songcn` should be at least `0.0.9`
2. download `bfosc`
    - `git clone https://github.com/hypergravity/bfosc.git`
3. revise the parameters in `bfosc_pipeline.py`
4. run it
    - `ipython bfosc_pipeline.py > bfosc_reduction_20201124.log`

5. For an example log file, click [**here**](https://github.com/hypergravity/bfosc/blob/main/E9G10/bfosc_reduction_20201124.log).

## E9G10
Currently, only E9+G10 configuration is tested.

## Orders

![](https://github.com/hypergravity/bfosc/tree/main/E9G10/template/E9G10_orders.png)