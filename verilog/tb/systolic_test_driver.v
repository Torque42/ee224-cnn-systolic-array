`timescale 1ns/1ps

module systolic_test_driver #(
    parameter integer MAX_M=676,
    parameter integer MAX_K=250,
    parameter integer MAX_N=128
);
    localparam integer W=8, ACC=24;
    reg clock=0, reset=0;
    reg [4*W-1:0] Ain=0, Bin=0;
    wire [4*W-1:0] Aout, Bout;
    wire [16*ACC-1:0] C;
    // Matrices are row-major; c_mem receives only DUT output values.
    integer a_mem [0:MAX_M*MAX_K-1];
    integer b_mem [0:MAX_K*MAX_N-1];
    integer c_mem [0:MAX_M*MAX_N-1];
    integer tile_cycles;

    always #5 clock=~clock;
    systolic_array #(.W(W), .W_accum(ACC)) dut (
        .clock(clock), .reset(reset), .Ain(Ain), .Bin(Bin),
        .A(Aout), .B(Bout), .C(C)
    );

    // Clear the previous tile before starting a new matrix product.
    task automatic clear_array;
        begin
            @(negedge clock);
            Ain=0;
            Bin=0;
            reset=1;
            #1;
            if (C!=={16*ACC{1'b0}}||Aout!==0||Bout!==0) begin
                $display("FAIL: Array asynchronous reset failed");
                $stop;
                $finish;
            end
            @(negedge clock);
            reset=0;
        end
    endtask

    task automatic matmul(input integer m, k, n);
        integer tr, tc, t, i, j, q, r, c, actual;
        integer expected [0:15];
        begin
            if (m<1||m>MAX_M||k<1||k>MAX_K||n<1||n>MAX_N) begin
                $display("FAIL: Unsupported dimensions %0dx%0dx%0d", m, k, n);
                $stop;
                $finish;
            end
            for (i=0; i<m*k; i=i+1)
                if (a_mem[i]<-128||a_mem[i]>127) begin
                    $display("FAIL: A[%0d] is outside signed int8", i);
                    $stop;
                    $finish;
                end
            for (i=0; i<k*n; i=i+1)
                if (b_mem[i]<-128||b_mem[i]>127) begin
                    $display("FAIL: B[%0d] is outside signed int8", i);
                    $stop;
                    $finish;
                end
            for (tr=0; tr<m; tr=tr+4) begin
                for (tc=0; tc<n; tc=tc+4) begin
                    clear_array();
                    for (i=0; i<16; i=i+1) expected[i]=0;
                    tile_cycles=0;

                    // Allow K+6 active edges, excluding reset and setup.
                    for (t=0; t<k+6; t=t+1) begin
                        @(negedge clock);
                        Ain=0;
                        Bin=0;
                        for (i=0; i<4; i=i+1) begin
                            // Skew row i and column i by i cycles.
                            q=t-i;
                            if (tr+i<m&&q>=0&&q<k)
                                Ain[i*W+:W]=a_mem[(tr+i)*k+q];
                            if (tc+i<n&&q>=0&&q<k)
                                Bin[i*W+:W]=b_mem[q*n+tc+i];
                        end
                        @(posedge clock);
                        #1; // Sample after nonblocking register updates settle.
                        tile_cycles=tile_cycles+1;
                        // Check partial sums against independent scalar products.
                        for (i=0; i<4; i=i+1) begin
                            for (j=0; j<4; j=j+1) begin
                                r=tr+i;
                                c=tc+j;
                                q=t-i-j;
                                if (r<m&&c<n&&q>=0&&q<k)
                                    expected[i*4+j]=expected[i*4+j]
                                        +a_mem[r*k+q]*b_mem[q*n+c];
                                actual=$signed(C[(i*4+j)*ACC+:ACC]);
                                if (actual!==expected[i*4+j]) begin
                                    $display("FAIL: MAC mismatch row=%0d col=%0d cycle=%0d got=%0d expected=%0d",
                                    r, c, tile_cycles, actual, expected[i*4+j]);
                                    $stop;
                                    $finish;
                                end
                            end
                        end
                        for (i=0; i<4; i=i+1) begin
                            q=t-i-3;
                            actual=0;
                            if (tr+i<m&&q>=0&&q<k)
                                actual=a_mem[(tr+i)*k+q];
                            if ($signed(Aout[i*W+:W])!==actual) begin
                                $display("FAIL: A boundary forwarding failed");
                                $stop;
                                $finish;
                            end
                            actual=0;
                            if (tc+i<n&&q>=0&&q<k)
                                actual=b_mem[q*n+tc+i];
                            if ($signed(Bout[i*W+:W])!==actual) begin
                                $display("FAIL: B boundary forwarding failed");
                                $stop;
                                $finish;
                            end
                        end
                    end
                    // Discard padded lanes at the matrix boundaries.
                    for (i=0; i<4; i=i+1)
                        for (j=0; j<4; j=j+1)
                            if (tr+i<m&&tc+j<n)
                                c_mem[(tr+i)*n+tc+j]=$signed(C[(i*4+j)*ACC+:ACC]);
                end
            end
        end
    endtask
endmodule
