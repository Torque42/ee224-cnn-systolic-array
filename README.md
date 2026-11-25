# Implementing Convolutional Neural Network Inference Through a 4×4 Systolic Array

EE224 Digital Systems — Course Project, Semester 3, 2026–2027.

This project implements a signed multiply-and-accumulate processing element and connects sixteen instances into a structural 4×4 systolic array. The array is used for matrix multiplication and for the convolutional and fully connected layers of an MNIST classifier. The [project report](report/Project%20Report%20-%20Ayush%20Agrawal%20-%20Rajveer%20Singh%20Choudhary%20-%20Atharva%20Airen.pdf) contains the derivations, numerical results, simulation waveforms, and source listings.

## Contributors

| Member | Roll number | Contributions |
|---|---|---|
| Ayush Agrawal | 25B1304 | Python implementation and processing-element Verilog |
| Rajveer Singh Choudhary | 25B1313 | Python implementation and systolic-array Verilog |
| Atharva Airen | 25B1207 | Verilog testbenches and report preparation |

## Part A: Processing element

At each rising clock edge, the processing element multiplies its two inputs, adds the product to the stored sum, and forwards the inputs to the next processing elements. Active-high asynchronous reset clears both forwarding registers and the accumulator.

The operands use an 8-bit signed two's complement representation, supporting integers in [−128,127]. A signed 16-bit wire stores the product, and a 24-bit accumulator stores the running sum. The longest dot product in the CNN contains 250 terms, giving the conservative bound

$$
|s|\leq250(128)^2=4{,}096{,}000.
$$

This is within the signed 24-bit range. The PE testbench checks positive, negative, and zero products, registered forwarding, asynchronous reset, and 250-term accumulation.

## Part B: Structural systolic array

The array connects sixteen PEs in a 4×4 grid. Each PE forwards its A input to the right and its B input downward. The operands move by one position per clock cycle, while the partial sum remains in the PE. For multiplication of a 4×K matrix A by a K×4 matrix B, PE (i,j) computes

$$
C_{ij}=\sum_{q=0}^{K-1}A_{iq}B_{qj}.
$$

Input row i and input column j are delayed by i and j clock edges so that matching operands arrive together. Counting the first sampling edge as edge 1, the last product reaches PE (3,3) on edge K+6.

| Matrix operation | Active clock edges | Behavioural simulation |
|---|---:|---|
| 4×4 by 4×4 | 10 | All sixteen entries match the reference |
| 4×8 by 8×4 | 14 | All sixteen entries match the reference |

These counts exclude reset and setup. Both results were verified with Icarus Verilog, including independent checks of the recorded waveforms. Additional cases cover negative operands, reset between products, partial output tiles, and the inner dimensions used by the CNN.

## Part C(i): MNIST CNN training

The classifier is trained on 60,000 grayscale MNIST images and evaluated on the separate 10,000-image test set. Pixel values are normalized to [0,1]. Both convolutions use 3×3 kernels, stride 1, and valid padding; both pooling layers use 2×2 windows with stride 2.

| Layer | Output shape | Parameters |
|---|---|---:|
| Input | 28×28×1 | 0 |
| Conv1 + ReLU | 26×26×5 | 50 |
| Max pool 1 | 13×13×5 | 0 |
| Conv2 + ReLU | 11×11×10 | 460 |
| Max pool 2 | 5×5×10 | 0 |
| Flatten and dropout | 250 | 0 |
| Dense1 + ReLU | 128 | 32,128 |
| Dense2 + softmax | 10 | 1,290 |
| **Total** | | **33,928** |

The Keras notebook uses Adam with a learning rate of 0.001, categorical cross-entropy, dropout of 0.45, and five training epochs. Final evaluation records 98.52% training accuracy and 98.33% test accuracy, both above the required 80%. The final epoch's training-curve value is approximately 96.27%, measured during training with dropout active and changing weights. The 98.52% result is measured afterward with fixed final weights and dropout disabled.

The notebook is the source of the reported training results. The separate PyTorch implementation uses the same layer structure and trains for 100 epochs.

## Part C(ii): Integer CNN inference

Keras weights and biases are extracted with `get_weights()`. Convolution kernels are reshaped into matrices whose columns represent output channels. Input patches are flattened in the same kernel-row, kernel-column, and channel order, allowing convolution to be computed as matrix multiplication.

Activations are scaled by 64 and weights by 128, then rounded and clipped to signed 8-bit values. Biases are scaled by 8192 to match the activation–weight products. The array accumulates products in 24-bit registers; the testbench adds biases using 32-bit integers. After each convolution and Dense1, an arithmetic right shift by seven restores the activation scale, and clipping to [0,127] applies ReLU and saturation. The final class is selected by the largest Dense2 logit.

The notebook prepares input matrices for all 100 images together. Each hidden-layer input is obtained by adding the preceding layer's bias, requantizing, applying ReLU, and pooling or flattening as required. The batch testbench loads these prepared matrices, executes all four products on the array, and compares every result with its golden value. It retains the simulated Dense2 outputs, adds the final biases, and selects the predicted digit by argmax.

| Layer | Batch input matrix | Padded weight matrix | Output tiles |
|---|---|---|---:|
| Conv1 | 67,600×9 | 9×8 | 33,800 |
| Conv2 | 12,100×45 | 45×12 | 9,075 |
| Dense1 | 100×250 | 250×128 | 800 |
| Dense2 | 100×128 | 128×12 | 75 |

The 19 matching integer export files used by the recorded run are included in `testdata/cnn/`. The manifest records the source commit, original and normalized file hashes, dimensions, and expected results. Only Windows line endings were normalized; all integer values are unchanged. The restored `tb_cnn.sv` retains the teammate's batch schedule and classification procedure, with portable file paths, input checks, and nonzero failure exits.

### Verified CNN results

The submitted testbench and frozen export reproduce the ModelSim summary in the report:

| Quantity | Result |
|---|---:|
| Images evaluated | 100, ten per digit |
| Classification accuracy | 99/100 (99%) |
| Matrix-product mismatches | 0 across 700,000 stored entries |
| Agreement with the matching NumPy predictions | 100/100 |
| Reported cycle count | 1,228,425 |

The cycle counter includes K+6 active edges and one reset edge for each tile; additional sampling/setup edges are not counted. The training curves describe the saved notebook training run. The frozen integer export comes from a separate training run of the same architecture, and its corresponding NumPy predictions also achieve 99/100.

## Source files

```text
python/
    cnn_part1.ipynb
    main.py
    model.py
    scripts.py
verilog/
    processing_element.v
    systolic_array.v
    tb/
        systolic_test_driver.sv
        tb_processing_element.sv
        tb_systolic_array.sv
        tb_cnn.sv
testdata/
    cnn/
        manifest.json
        *.txt
report/
    Project Report - Ayush Agrawal - Rajveer Singh Choudhary - Atharva Airen.pdf
```

`ProcessingElement` is defined once in `processing_element.v`. The structural array instantiates that module sixteen times. Only the compiled PDF is tracked under `report/`; the LaTeX sources and figure-generation files are retained locally.

## Running the implementation

The hardware checks require Icarus Verilog (`iverilog` and `vvp`). Run from the repository root:

```sh
mkdir -p build
iverilog -g2012 -Wall -s tb_processing_element -o build/pe.vvp verilog/processing_element.v verilog/tb/tb_processing_element.sv
vvp build/pe.vvp +VCD
iverilog -g2012 -Wall -s tb_systolic_array -o build/array.vvp verilog/processing_element.v verilog/systolic_array.v verilog/tb/systolic_test_driver.sv verilog/tb/tb_systolic_array.sv
vvp build/array.vvp +VCD
iverilog -g2012 -Wall -s tb_cnn -o build/cnn.vvp verilog/processing_element.v verilog/systolic_array.v verilog/tb/tb_cnn.sv
vvp build/cnn.vvp
```

All three benches print PASS on success and exit with an error on invalid data or a failed comparison. CNN inference uses `testdata/cnn/` by default, so no Python installation or retraining is required to reproduce the fixed test. `+DATA_DIR=/path/to/export` selects another export directory. `+VCD` enables optional waveform recording; CNN traces can be large.

For ModelSim/Questa:

```tcl
vlib work
vlog verilog/processing_element.v verilog/systolic_array.v
vlog -sv verilog/tb/systolic_test_driver.sv verilog/tb/tb_processing_element.sv verilog/tb/tb_systolic_array.sv verilog/tb/tb_cnn.sv
vsim -c work.tb_processing_element -do "run -all; quit -f"
vsim -c work.tb_systolic_array -do "run -all; quit -f"
vsim -c work.tb_cnn -do "run -all; quit -f"
```

To train and export a new model, run the three cells in `python/cnn_part1.ipynb` in order with TensorFlow, NumPy, Matplotlib, and a notebook environment. The export is written to `modelsim_data/` relative to the notebook working directory. A new stochastic training run can produce a different accuracy; its matrices, biases, labels, and predictions must be used together. The separate PyTorch implementation requires PyTorch, torchvision, and Matplotlib.
