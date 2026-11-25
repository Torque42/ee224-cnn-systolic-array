`timescale 1ns/1ps

module tb_cnn;

    parameter W=8;
    parameter AW=24;
    parameter NIMG=100;

    parameter DIR="testdata/cnn";
    localparam PATH_BYTES=1024;
    reg [8*PATH_BYTES-1:0] data_dir;
    integer ignored, digit;
    integer histogram [0:9];

    reg clock, reset;
    reg [4*W-1:0] Ain, Bin;
    wire [16*AW-1:0] C;

    // Flattened batch matrices and expected products are stored row-major.
    integer A_mat [0:700000];
    integer B_mat [0:40000];
    integer expected [0:600000];

    integer logits [0:NIMG*16-1];
    integer bias2 [0:15];
    integer labels [0:NIMG-1];
    integer expect_pred [0:NIMG-1];

    integer M, K, N;
    integer errors, total_errors;
    integer cycles, total_cycles;

    systolic_array #(.W(W), .W_accum(AW)) DUT (
        .clock(clock),
        .reset(reset),
        .Ain(Ain),
        .Bin(Bin),
        .C(C)
    );

    initial begin
        clock=0;
        forever #5 clock=~clock;
    end

    // Read exactly the declared matrix size; no trailing-EOF probe is needed.
    task load_file;
        input [8*(PATH_BYTES+32)-1:0] fname;
        input integer count;
        input integer which;

        integer fd, i, value, rc;

        begin
            fd=$fopen(fname, "r");

            if(fd==0) begin
                $display("ERROR: Cannot open file %0s", fname);
                $display("FAIL: CNN simulation failed");
                $stop;
                $finish;
            end

            for(i=0; i<count; i=i+1) begin
                rc=$fscanf(fd, "%d", value);

                if(rc!=1) begin
                    $display("ERROR while reading file at index %0d", i);
                    $fclose(fd);
                    $display("FAIL: CNN simulation failed");
                    $stop;
                    $finish;
                end

                if(which!=2&&(value<-128||value>127)) begin
                    $display("FAIL: Operand outside signed int8 range in %0s", fname);
                    $stop;
                    $finish;
                end
                if(which==0)
                    A_mat[i]=value;
                else if(which==1)
                    B_mat[i]=value;
                else
                    expected[i]=value;
            end

            $fclose(fd);
        end
    endtask

    task run_layer;
        input [8*16-1:0] name;

        integer mt, nt, t, i, j, index;
        integer got;
        integer fd, rc;

        reg [8*(PATH_BYTES+32)-1:0] fname;

        begin
            $sformat(fname, "%0s/%0s_dims.txt", data_dir, name);

            fd=$fopen(fname, "r");

            if(fd==0) begin
                $display("ERROR: Cannot open dimensions file");
                $display("FAIL: CNN simulation failed");
                $stop;
                $finish;
            end

            rc=$fscanf(fd, "%d %d %d", M, K, N);
            $fclose(fd);

            if(rc!=3) begin
                $display("ERROR: Invalid dimensions file");
                $display("FAIL: CNN simulation failed");
                $stop;
                $finish;
            end

            if(M<=0||K<=0||N<=0||M%4!=0||N%4!=0||
               M*K>700001||K*N>40001||M*N>600001) begin
                $display("FAIL: Unsupported dimensions for %0s", name);
                $stop;
                $finish;
            end
            if(name=="dense2"&&(M!=NIMG||N<10||N>16)) begin
                $display("FAIL: Dense2 dimensions do not match the classifier");
                $stop;
                $finish;
            end

            $sformat(fname, "%0s/%0s_A.txt", data_dir, name);
            load_file(fname, M*K, 0);

            $sformat(fname, "%0s/%0s_B.txt", data_dir, name);
            load_file(fname, K*N, 1);

            $sformat(fname, "%0s/%0s_golden.txt", data_dir, name);
            load_file(fname, M*N, 2);

            errors=0;
            cycles=0;

            for(mt=0; mt<M/4; mt=mt+1) begin
                for(nt=0; nt<N/4; nt=nt+1) begin
                    @(negedge clock);

                    reset=1;
                    Ain=0;
                    Bin=0;

                    cycles=cycles+1; // Count one reset edge for this output tile.

                    // Skew each input lane, then flush the array through edge K+6.
                    for(t=0; t<K+6; t=t+1) begin
                        @(negedge clock);

                        reset=0;

                        for(i=0; i<4; i=i+1) begin
                            index=t-i;

                            if(index>=0&&index<K)
                                Ain[i*W+:W]=
                                    A_mat[(mt*4+i)*K+index];
                            else
                                Ain[i*W+:W]=0;
                        end

                        for(j=0; j<4; j=j+1) begin
                            index=t-j;

                            if(index>=0&&index<K)
                                Bin[j*W+:W]=
                                    B_mat[index*N+nt*4+j];
                            else
                                Bin[j*W+:W]=0;
                        end

                        cycles=cycles+1;
                    end

                    @(negedge clock);

                    for(i=0; i<4; i=i+1) begin
                        for(j=0; j<4; j=j+1) begin
                            // Check every DUT output and retain Dense2 logits for classification.
                            got=$signed(C[(i*4+j)*AW+:AW]);

                            index=(mt*4+i)*N+nt*4+j;

                            if(name=="dense2") begin
                                logits[(mt*4+i)*16+nt*4+j]=got;
                            end

                            if(got!==expected[index]) begin
                                errors=errors+1;

                                if(errors<=5) begin
                                    $display(
                                        "ERROR in %0s at tile(%0d,%0d), C[%0d][%0d]: Got=%0d Expected=%0d",
                                        name, mt, nt, i, j,
                                        got, expected[index]
                                    );
                                end
                            end
                        end
                    end
                end
            end

            total_errors=total_errors+errors;
            total_cycles=total_cycles+cycles;

            $display("----------------------------------------------");
            $display("Layer: %0s", name);
            $display("Matrix A: %0d x %0d", M, K);
            $display("Matrix B: %0d x %0d", K, N);
            $display("Number of tiles: %0d", (M/4)*(N/4));
            $display("Clock cycles: %0d", cycles);
            $display("Mismatches: %0d", errors);
            $display("----------------------------------------------");
        end
    endtask

    initial begin
        clock=0;
        reset=1;
        Ain=0;
        Bin=0;

        data_dir=DIR;
        ignored=$value$plusargs("DATA_DIR=%s", data_dir);
        if($test$plusargs("VCD")) begin
            $dumpfile("tb_cnn.vcd");
            $dumpvars(0, tb_cnn);
        end
        for(digit=0; digit<10; digit=digit+1) histogram[digit]=0;

        total_errors=0;
        total_cycles=0;

        // Each layer consumes its exported matrices; all four products are verified.
        run_layer("conv1");
        run_layer("conv2");
        run_layer("dense1");
        run_layer("dense2");

        begin : classification

            integer i, j;
            integer fd, rc;
            integer best, bestv;
            integer correct, agree;

            reg [8*(PATH_BYTES+32)-1:0] fname;

            $sformat(fname, "%0s/b_dense2.txt", data_dir);

            fd=$fopen(fname, "r");

            if(fd==0) begin
                $display("ERROR: Cannot open dense2 bias file");
                $display("FAIL: CNN simulation failed");
                $stop;
                $finish;
            end

            for(i=0; i<10; i=i+1) begin
                rc=$fscanf(fd, "%d", bias2[i]);
                if(rc!=1) begin
                    $display("FAIL: Missing Dense2 bias at index %0d", i);
                    $stop;
                    $finish;
                end
            end

            $fclose(fd);

            $sformat(fname, "%0s/test_labels_100.txt", data_dir);

            fd=$fopen(fname, "r");

            if(fd==0) begin
                $display("ERROR: Cannot open labels file");
                $display("FAIL: CNN simulation failed");
                $stop;
                $finish;
            end

            for(i=0; i<NIMG; i=i+1) begin
                rc=$fscanf(fd, "%d", labels[i]);
                if(rc!=1||labels[i]<0||labels[i]>9) begin
                    $display("FAIL: Invalid label at image %0d", i);
                    $stop;
                    $finish;
                end
                histogram[labels[i]]=histogram[labels[i]]+1;
            end
            for(i=0; i<10; i=i+1)
                if(histogram[i]!=NIMG/10||NIMG%10!=0) begin
                    $display("FAIL: Unbalanced label count for digit %0d", i);
                    $stop;
                    $finish;
                end

            $fclose(fd);

            $sformat(fname, "%0s/expected_pred_100.txt", data_dir);

            fd=$fopen(fname, "r");

            if(fd==0) begin
                $display("ERROR: Cannot open predictions file");
                $display("FAIL: CNN simulation failed");
                $stop;
                $finish;
            end

            for(i=0; i<NIMG; i=i+1) begin
                rc=$fscanf(fd, "%d", expect_pred[i]);
                if(rc!=1||expect_pred[i]<0||expect_pred[i]>9) begin
                    $display("FAIL: Invalid reference prediction at image %0d", i);
                    $stop;
                    $finish;
                end
            end

            $fclose(fd);

            correct=0;
            agree=0;

            for(i=0; i<NIMG; i=i+1) begin
                // Add the final bias and use the first maximum, as in NumPy argmax.
                best=0;
                bestv=logits[i*16]+bias2[0];

                for(j=1; j<10; j=j+1) begin
                    if(logits[i*16+j]+bias2[j]>bestv) begin
                        bestv=logits[i*16+j]+bias2[j];
                        best=j;
                    end
                end

                if(best==labels[i])
                    correct=correct+1;

                if(best==expect_pred[i])
                    agree=agree+1;

                $display(
                    "Image %0d: Label=%0d Predicted=%0d",
                    i, labels[i], best
                );
            end

            $display("\n================================================");

            $display(
                "Total matrix multiplication mismatches: %0d",
                total_errors
            );

            $display(
                "Total clock cycles: %0d",
                total_cycles
            );

            $display(
                "Systolic array test accuracy: %0d / %0d (%0d %%)",
                correct, NIMG, correct*100/NIMG
            );

            $display(
                "Agreement with NumPy model: %0d / %0d",
                agree, NIMG
            );

            $display("================================================");
            if(total_errors!=0||agree!=NIMG) begin
                $display("FAIL: CNN results differ from the integer reference");
                $stop;
                $finish;
            end
        end

        $display("PASS: all CNN matrix products and predictions match the integer reference");
        $finish;
    end
endmodule
