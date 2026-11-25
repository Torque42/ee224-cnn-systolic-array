# Implementing Convolutional Neural Network Inference Through a 4×4 Systolic Array

EE224 Digital Systems — Course Project, Semester 1, 2026–2027.

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

| Layer | Input matrix | Weight matrix | Output tiles per image |
|---|---|---|---:|
| Conv1 | 676×9 | 9×5 | 338 |
| Conv2 | 121×45 | 45×10 | 93 |
| Dense1 | 1×250 | 250×128 | 32 |
| Dense2 | 1×128 | 128×10 | 3 |

Each multiplication is divided into 4×4 output tiles. The driver streams all K terms for each tile, fills unused boundary lanes with zeros, and resets the accumulators before the next tile. Bias addition, activation, pooling, flattening, and classification are performed in the testbench. The per-image CNN testbench constructs subsequent layers from DUT results and uses exported reference matrices only for comparison.

### Recorded CNN results

The report includes a completed ModelSim run on 100 MNIST test images, with ten images from each digit class.

| Quantity | ModelSim result |
|---|---:|
| Classification accuracy | 99/100 (99%) |
| Matrix-product mismatches against NumPy | 0 |
| Agreement with the corresponding NumPy integer model | 100/100 |
| Total reported clock cycles, all layers | 1,228,425 |

These measurements belong to the batched `tb_cnn` run shown in the report. Its cycle count includes one reset cycle per output tile and differs from the per-image schedule of `tb_cnn_inference.sv`. The older notebook output records 98/100 for a separate integer-inference run; it should not be treated as the NumPy reference for the 99/100 ModelSim experiment.

### Current verification status

The PE and array testbenches pass, and all three testbench tops compile. The current per-image CNN testbench has an end-of-file handling issue under Icarus Verilog 12: `close_data` treats a return value of zero after trailing whitespace as an input error, even when the file has ended normally. This check must be corrected before using that bench for a full CNN run. The issue is separate from the completed batched ModelSim run documented above.

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
        tb_cnn_inference.sv
report/
    Project Report - Ayush Agrawal - Rajveer Singh Choudhary - Atharva Airen.pdf
```

`ProcessingElement` is defined once in `processing_element.v`. The structural array instantiates that module sixteen times. Only the compiled PDF is tracked under `report/`; the LaTeX sources and figure-generation files are retained locally.

## Running the implementation

The Keras notebook requires TensorFlow, NumPy, Matplotlib, and a notebook environment. Run its first three cells in order to train, quantize, and export a consistent set of inputs and references. With `python/` as the notebook working directory, the files are written to `python/modelsim_data/`. The alternate PyTorch implementation requires PyTorch, torchvision, and Matplotlib.

With Icarus Verilog installed, run the PE and array checks from the repository root:

```sh
mkdir -p build
iverilog -g2012 -Wall -s tb_processing_element -o build/pe.vvp verilog/processing_element.v verilog/tb/tb_processing_element.sv
vvp build/pe.vvp +VCD
iverilog -g2012 -Wall -s tb_systolic_array -o build/array.vvp verilog/processing_element.v verilog/systolic_array.v verilog/tb/systolic_test_driver.sv verilog/tb/tb_systolic_array.sv
vvp build/array.vvp +VCD
```

The benches stop on a mismatch and print PASS messages on success. `+VCD` enables waveform recording. To compile the per-image CNN bench:

```sh
iverilog -g2012 -Wall -s tb_cnn_inference -o build/cnn.vvp verilog/processing_element.v verilog/systolic_array.v verilog/tb/systolic_test_driver.sv verilog/tb/tb_cnn_inference.sv
```

Once the EOF check has been corrected and the notebook exports are available, the runtime accepts `+DATA_DIR=python/modelsim_data`. All images, weights, biases, labels, reference predictions, and golden matrices must come from the same export.
