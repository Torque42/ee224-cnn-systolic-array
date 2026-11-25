`timescale 1ns/1ps

module tb_cnn_inference;
    systolic_test_driver driver();
    string data_dir="python/modelsim_data";
    integer images [0:100*784-1];
    integer labels [0:99], predictions [0:99], histogram [0:9];
    integer w1 [0:44], w2 [0:449], w3 [0:31999], w4 [0:1279];
    integer b1 [0:4], b2 [0:9], b3 [0:127], b4 [0:9];
    integer pool1 [0:844], pool2 [0:249], dense1 [0:127];
    integer golden1, golden2, golden3, golden4;
    integer fd, i, sample_id, correct, prediction, best, score, ignored;

    function automatic integer open_data(input string name);
        integer handle;
        begin
            handle=$fopen({data_dir, "/", name}, "r");
            if (!handle) $fatal(1, "Cannot open %s/%s; run notebook cells 0-2 first", data_dir, name);
            open_data=handle;
        end
    endfunction

    function automatic integer read_value(input integer handle, lo, hi);
        integer value, status;
        begin
            status=$fscanf(handle, "%d", value);
            if (status!=1) $fatal(1, "Missing or malformed integer in input file (handle %0d)", handle);
            if (value<lo||value>hi)
                $fatal(1, "Integer %0d outside expected range [%0d,%0d]", value, lo, hi);
            read_value=value;
        end
    endfunction

    task automatic close_data(input integer handle);
        integer status, value;
        begin
            status=$fscanf(handle, "%d", value);
            if (status!=-1) $fatal(1, "Unexpected trailing data in input file (handle %0d)", handle);
            $fclose(handle);
        end
    endtask

    // Restore the activation scale, then apply ReLU and int8 saturation.
    function automatic integer activate(input integer value);
        integer shifted;
        begin

            shifted=value>>>7;
            if (shifted<0) activate=0;
            else if (shifted>127) activate=127;
            else activate=shifted;
        end
    endfunction

    task automatic check_golden(input integer handle, m, n, input string layer);
        integer r, c, expected, actual, padded_n;
        begin
            // Notebook goldens pad output columns to a multiple of four.
            padded_n=((n+3)/4)*4;
            for (r=0; r<m; r=r+1) begin
                for (c=0; c<padded_n; c=c+1) begin
                    expected=read_value(handle,-8388608, 8388607);
                    actual=0;
                    if (c<n) actual=driver.c_mem[r*n+c];
                    if (actual!==expected)
                        $fatal(1, "%s NumPy mismatch image=%0d row=%0d col=%0d got=%0d expected=%0d",
                            layer, sample_id, r, c, actual, expected);
                end
            end
        end
    endtask

    task automatic conv1;
        integer r, c, kr, kc, ch, dy, dx, value, maximum;
        begin

            // Match the kernel-row, kernel-column, channel order used by Keras.
            for (r=0; r<26; r=r+1)
                for (c=0; c<26; c=c+1)
                    for (kr=0; kr<3; kr=kr+1)
                        for (kc=0; kc<3; kc=kc+1)
                            driver.a_mem[(r*26+c)*9+kr*3+kc]=images[sample_id*784+(r+kr)*28+c+kc];
            for (r=0; r<45; r=r+1) driver.b_mem[r]=w1[r];
            driver.matmul(676, 9, 5);
            check_golden(golden1, 676, 5, "conv1");
            for (r=0; r<13; r=r+1)
                for (c=0; c<13; c=c+1)
                    for (ch=0; ch<5; ch=ch+1) begin
                        maximum=0;
                        for (dy=0; dy<2; dy=dy+1)
                            for (dx=0; dx<2; dx=dx+1) begin
                                value=activate(driver.c_mem[((2*r+dy)*26+2*c+dx)*5+ch]+b1[ch]);
                                if (value>maximum) maximum=value;
                            end
                        pool1[(r*13+c)*5+ch]=maximum;
                    end
        end
    endtask

    task automatic conv2;
        integer r, c, kr, kc, ch, dy, dx, value, maximum;
        begin
            // Build this layer from the preceding DUT results after pooling.
            for (r=0; r<11; r=r+1)
                for (c=0; c<11; c=c+1)
                    for (kr=0; kr<3; kr=kr+1)
                        for (kc=0; kc<3; kc=kc+1)
                            for (ch=0; ch<5; ch=ch+1)
                                driver.a_mem[(r*11+c)*45+(kr*3+kc)*5+ch]=pool1[((r+kr)*13+c+kc)*5+ch];
            for (r=0; r<450; r=r+1) driver.b_mem[r]=w2[r];
            driver.matmul(121, 45, 10);
            check_golden(golden2, 121, 10, "conv2");

            for (r=0; r<5; r=r+1)
                for (c=0; c<5; c=c+1)
                    for (ch=0; ch<10; ch=ch+1) begin
                        maximum=0;
                        for (dy=0; dy<2; dy=dy+1)
                            for (dx=0; dx<2; dx=dx+1) begin
                                value=activate(driver.c_mem[((2*r+dy)*11+2*c+dx)*10+ch]+b2[ch]);
                                if (value>maximum) maximum=value;
                            end
                        pool2[(r*5+c)*10+ch]=maximum;
                    end
        end
    endtask

    task automatic fully_connected;
        integer r;
        begin
            for (r=0; r<250; r=r+1) driver.a_mem[r]=pool2[r];
            for (r=0; r<32000; r=r+1) driver.b_mem[r]=w3[r];
            driver.matmul(1, 250, 128);
            check_golden(golden3, 1, 128, "dense1");
            for (r=0; r<128; r=r+1) dense1[r]=activate(driver.c_mem[r]+b3[r]);
            for (r=0; r<128; r=r+1) driver.a_mem[r]=dense1[r];
            for (r=0; r<1280; r=r+1) driver.b_mem[r]=w4[r];
            driver.matmul(1, 128, 10);
            check_golden(golden4, 1, 10, "dense2");
        end
    endtask

    initial begin
        ignored=$value$plusargs("DATA_DIR=%s", data_dir);
        if ($test$plusargs("VCD")) begin
            $dumpfile("tb_cnn_inference.vcd");
            $dumpvars(0, driver.dut);
        end
        fd=open_data("test_images_100.txt");
        for (i=0; i<100*784; i=i+1) images[i]=read_value(fd, 0, 64);
        close_data(fd);
        for (i=0; i<10; i=i+1) histogram[i]=0;
        fd=open_data("test_labels_100.txt");
        for (i=0; i<100; i=i+1) begin
            labels[i]=read_value(fd, 0, 9);
            histogram[labels[i]]=histogram[labels[i]]+1;
        end
        close_data(fd);
        // Require a balanced set with ten test images per digit.
        for (i=0; i<10; i=i+1)
            if (histogram[i]!=10) $fatal(1, "Expected exactly 10 samples of digit %0d", i);
        fd=open_data("expected_pred_100.txt");
        for (i=0; i<100; i=i+1) predictions[i]=read_value(fd, 0, 9);
        close_data(fd);
        fd=open_data("w_conv1.txt");
        for (i=0; i<45; i=i+1) w1[i]=read_value(fd,-128, 127);
        close_data(fd);
        fd=open_data("w_conv2.txt");
        for (i=0; i<450; i=i+1) w2[i]=read_value(fd,-128, 127);
        close_data(fd);
        fd=open_data("w_dense1.txt");
        for (i=0; i<32000; i=i+1) w3[i]=read_value(fd,-128, 127);
        close_data(fd);
        fd=open_data("w_dense2.txt");
        for (i=0; i<1280; i=i+1) w4[i]=read_value(fd,-128, 127);
        close_data(fd);

        fd=open_data("b_conv1.txt");
        for (i=0; i<5; i=i+1) b1[i]=read_value(fd,-1073741824, 1073741823);
        close_data(fd);
        fd=open_data("b_conv2.txt");
        for (i=0; i<10; i=i+1) b2[i]=read_value(fd,-1073741824, 1073741823);
        close_data(fd);
        fd=open_data("b_dense1.txt");
        for (i=0; i<128; i=i+1) b3[i]=read_value(fd,-1073741824, 1073741823);
        close_data(fd);
        fd=open_data("b_dense2.txt");
        for (i=0; i<10; i=i+1) b4[i]=read_value(fd,-1073741824, 1073741823);
        close_data(fd);
        golden1=open_data("conv1_golden.txt");
        golden2=open_data("conv2_golden.txt");
        golden3=open_data("dense1_golden.txt");
        golden4=open_data("dense2_golden.txt");
        correct=0;
        for (sample_id=0; sample_id<100; sample_id=sample_id+1) begin
            conv1();
            conv2();
            fully_connected();
            prediction=0;
            best=driver.c_mem[0]+b4[0];
            for (i=1; i<10; i=i+1) begin
                score=driver.c_mem[i]+b4[i];
                if (score>best) begin
                    best=score;
                    prediction=i;
                end
            end
            if (prediction!=predictions[sample_id])
                $fatal(1, "Prediction mismatch image=%0d got=%0d NumPy=%0d", sample_id, prediction, predictions[sample_id]);
            if (prediction==labels[sample_id]) correct=correct+1;
            $display("Image %0d: label=%0d prediction=%0d", sample_id, labels[sample_id], prediction);
        end
        close_data(golden1);
        close_data(golden2);
        close_data(golden3);
        close_data(golden4);
        $display("PASS: all four DUT matrix operations match NumPy for every image");
        $display("DUT-based MNIST accuracy: %0d / 100 (%0d%%)", correct, correct);
        $finish;
    end
endmodule
