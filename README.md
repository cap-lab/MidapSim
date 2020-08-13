# A Novel CNN Accelerator That Enables Fully-Pipelined Execution of Layers

By [Donghyun Kang](http://iris.snu.ac.kr/xe/kangdongh), [Jintaek Kang](http://iris.snu.ac.kr/xe/taek0208), [Soonhoi Ha](https://peace.snu.ac.kr/sha/).

### Introduction

MIDAP, Memory In the Datapath Architecture Processor, features bus-free multi-bank on-chip memory architecture. For more details, please refer to our [ICCD Paper](https://ieeexplore.ieee.org/document/8988663).

### Citing MIDAP

Please cite MIDAP in your publications if it helps your research:

    @inproceedings{kang2019novel,
        title = {A Novel Convolutional Neural Network Accelerator That Enables Fully-Pipelined Execution of Layers},
        author = { D. {Kang} and J. {Kang} and H. {Kwon} and H. {Park} and S. {Ha} },
        booktitle = { 2019 IEEE 37th International Conference on Computer Design (ICCD) },
        year = {2019},
        pages = {698--701},
    }

This repository includes MIDAP Compiler & MIDAP Simulator

--Midap Simulator can be excuted with dedicated simulator instruction, please see data_structure/simulator_instruction.py

--Midap Compiler code will be refactored & modulized soon..

### How to install?

1. Get the code.
    ```Shell
    git clone https://github.com/cap-lab/MIDAPSim.git
    cd MIDAPSim
    ```

2. Install requirements.
    ```Shell
    pip install -r requirements.txt
    ```

3. Run the code at the root directory
    ```Shell
    python test.py -n test
    ```

### How to simulate?

You can define your own network with ModelBuilder
please refer test code with models/model_builder.py and models/examples.py 

1. you can use [TestWrapper class](test_wrapper.py) for easy simulation.

2. Please refer test.py for more information
    ```Shell
    python test.py -h
    python test.py -n test # Test MidapSim
    ```

### Tensor Virtualization

1. We proposed a novel virtualization technique [DAC Paper] _( it will be available soon)_ and tested it via MIDAPSim.

2. Tensor virtualization technique is applied to Concat, Upsample, TransposedConv (UpsampleZero + Conv) Layers.
