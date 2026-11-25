# EE224: CNN inference through a systolic array

Course project for EE224 Digital Systems, Semester 1, 2026–2027. The project implements a processing element, a structural 4×4 systolic array, and integer CNN inference for MNIST, following the supplied course question paper.

## Team

| Member | Roll number | Contributions |
|---|---|---|
| Ayush Agrawal | 25B1304 | Python code and processing-element Verilog |
| Rajveer Singh Choudhary | 25B1313 | Python code and systolic-array Verilog |
| Atharva Airen | 25B1207 | Verilog testbenches and report |

## Question paper and implementation

| Part | Requirement | Implementation |
|---|---|---|
| A — 5 marks | Registered input forwarding and multiply–accumulate PE; behavioural verification | `verilog/processing_element.v`, `verilog/tb/tb_processing_element.sv` |
| B — 10 marks | Structural 4×4 array; multiply 4×4 by 4×4 and 4×8 by 8×4; determine minimum cycles | `verilog/systolic_array.v`, `verilog/tb/tb_systolic_array.sv` |
| C(i) — within 5 marks for C | Two convolution/max-pool stages; MNIST training on 60,000 images and testing on 10,000; above 80% accuracy on each; accuracy plots | `python/cnn_part1.ipynb`; alternate PyTorch implementation in `python/main.py`, `model.py`, and `scripts.py` |
| C(ii) | Integer weights; convolution and dense layers as array matrix products; behavioural verification and accuracy on about 100 images equally distributed over ten digits | Notebook quantization/export cells, `verilog/tb/tb_cnn_inference.sv`, and supplied ModelSim evidence |

The PE forwards both operands on each rising edge and accumulates their signed product. Active-high asynchronous reset clears its outputs and accumulator. The default operands are signed 8-bit values; accumulators are 24 bits. `ProcessingElement` is defined only in `processing_element.v`; the array instantiates it sixteen times.

Rows and columns enter the array with staggered starts. A full output tile with inner dimension K completes in K+6 active clock edges, excluding reset and setup: 10 edges for 4×4 by 4×4 and 14 for 4×8 by 8×4. The shared test driver handles larger matrices using 4×4 output tiles and zero-fills unused boundary lanes.

## Files

```text
README.md
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
    Project Report.pdf
```

The report contains answers to each question, derivations, accuracy plots, numerical examples, waveforms, supplied ModelSim screenshots, and source listings. It includes all three names and roll numbers on the first page. The PDF is available as [Project Report.pdf](report/Project%20Report.pdf). Only the compiled report is tracked; its LaTeX sources, figures, and report-generation files are kept locally.

## Recorded results

| Check | Result | Evidence |
|---|---|---|
| PE forwarding, signed MAC, asynchronous reset, 250-term accumulation | Passed | Local Icarus simulation |
| 4×4 by 4×4 | All 16 entries match; 10 active edges | Local simulation and VCD checks |
| 4×8 by 8×4 | All 16 entries match; 14 active edges | Local simulation and VCD checks |
| Floating-point training accuracy | 98.52% | Saved Keras notebook evaluation |
| Floating-point test accuracy | 98.33% | Saved Keras notebook evaluation |
| Archived Python integer accuracy | 98/100 | Older saved notebook output |
| Systolic-array CNN accuracy | 99/100 (99%) | Teammate-provided final ModelSim transcript |
| Matrix-product mismatches | 0 | Supplied ModelSim transcript |
| Agreement with corresponding NumPy integer model | 100/100 | Supplied ModelSim transcript |
| Total cycles, all layers | 1,228,425 | Supplied ModelSim transcript |

The archived 98/100 notebook output and the 99/100 ModelSim result come from separate run artifacts. The latter agrees with its corresponding NumPy model on all 100 predictions. Its batched schedule differs from the local per-image testbench. The local CNN testbench has been compiled; the supplied 99% result is evidence from the teammate's run.

## Run A and B with Icarus Verilog

Install Icarus Verilog with its `iverilog` compiler and `vvp` runtime. Run these commands from the repository root:

```sh
mkdir -p build
iverilog -g2012 -Wall -s tb_processing_element -o build/pe.vvp verilog/processing_element.v verilog/tb/tb_processing_element.sv
vvp build/pe.vvp +VCD
iverilog -g2012 -Wall -s tb_systolic_array -o build/array.vvp verilog/processing_element.v verilog/systolic_array.v verilog/tb/systolic_test_driver.sv verilog/tb/tb_systolic_array.sv
vvp build/array.vvp +VCD
```

The benches stop on a mismatch and print PASS messages on success. `+VCD` writes `tb_processing_element.vcd` or `tb_systolic_array.vcd`; omit it when waveforms are unnecessary. The array bench also checks signed values, accumulator reset, partial tiles, and the CNN inner dimensions.

## Prepare and run C

The Keras notebook requires TensorFlow, NumPy, Matplotlib, and a notebook environment. Run its first three cells in order. With `python/` as the notebook working directory, the export directory is `python/modelsim_data/`.

The architecture is Conv(5, 3×3), max pool(2×2), Conv(10, 3×3), max pool(2×2), flatten, dropout(0.45), Dense(128), and Dense(10). Training uses Adam at 0.001 and categorical cross-entropy. The saved Keras run uses five epochs. The alternate PyTorch entry point trains for 100 epochs and requires PyTorch, torchvision, and Matplotlib.

Weights are signed int8 scaled by 128; input pixels are scaled by 64; biases are scaled by 8192. Hidden activations use an arithmetic shift by seven, ReLU, and saturation to [0,127]. Valid pooling and NHWC flattening match the Keras model.

The first convolution becomes (676×9)·(9×5), the second (121×45)·(45×10), and the dense layers (1×250)·(250×128) and (1×128)·(128×10), per image. All four matrix products execute on the same array. Bias addition, activation, pooling, and argmax execute in the testbench, with subsequent layers built from DUT results.

```sh
mkdir -p build
iverilog -g2012 -Wall -s tb_cnn_inference -o build/cnn.vvp verilog/processing_element.v verilog/systolic_array.v verilog/tb/systolic_test_driver.sv verilog/tb/tb_cnn_inference.sv
vvp build/cnn.vvp +DATA_DIR=python/modelsim_data
```

The CNN bench needs raw images, all weights and biases, labels, reference predictions, and four golden matrix files from the same export. Generate all required inputs with the updated notebook before running this bench. The bench requires ten images per digit, checks matrix results against the reference, and reports accuracy against ground-truth labels.

## ModelSim / Questa

From the repository root:

```tcl
vlib work
vlog verilog/processing_element.v verilog/systolic_array.v
vlog -sv verilog/tb/systolic_test_driver.sv verilog/tb/tb_processing_element.sv verilog/tb/tb_systolic_array.sv verilog/tb/tb_cnn_inference.sv
vsim -c work.tb_processing_element -do "run -all; quit -f"
vsim -c work.tb_systolic_array -do "run -all; quit -f"
vsim -c work.tb_cnn_inference +DATA_DIR=python/modelsim_data -do "run -all; quit -f"
```
