`timescale 1ns/1ps
module tb_systolic_array;
    systolic_test_driver #(.MAX_M(8), .MAX_K(250), .MAX_N(12)) driver();

    task automatic run_case(input integer m, k, n, seed, zero_inputs);
        integer r, c, q, expected;
        begin
            for (r=0; r<m; r=r+1)
                for (q=0; q<k; q=q+1)
                    driver.a_mem[r*k+q]=zero_inputs ? 0 : ((r*29+q*7+seed)%256)-128;
            for (q=0; q<k; q=q+1)
                for (c=0; c<n; c=c+1)
                    driver.b_mem[q*n+c]=((q*11+c*31+seed*3)%256)-128;
            driver.matmul(m, k, n);
            for (r=0; r<m; r=r+1) begin
                for (c=0; c<n; c=c+1) begin
                    // Scalar reference calculation, independent of input skew.
                    expected=0;
                    for (q=0; q<k; q=q+1)
                        expected=expected+driver.a_mem[r*k+q]*driver.b_mem[q*n+c];
                    if (driver.c_mem[r*n+c]!==expected)
                        $fatal(1, "Final matrix mismatch at (%0d,%0d)", r, c);
                end
            end
            $display("PASS: %0dx%0d * %0dx%0d; %0d active cycles per tile (reset excluded)",
                m, k, k, n, driver.tile_cycles);
        end
    endtask

    initial begin
        if ($test$plusargs("VCD")) begin
            $dumpfile("tb_systolic_array.vcd");
            $dumpvars(0, tb_systolic_array);
        end
        run_case(4, 4, 4, 1, 0); // B(i): 10 active edges.
        run_case(4, 8, 4, 9, 0); // B(ii): 14 active edges.
        // Further cases check reset, partial tiles, and CNN inner dimensions.
        run_case(4, 8, 4, 1, 1);
        run_case(5, 9, 5, 13, 0);
        run_case(4, 45, 10, 3, 0);
        run_case(1, 250, 7, 5, 0);
        run_case(1, 128, 10, 7, 0);
        $display("PASS: all array checks");
        $finish;
    end
endmodule
