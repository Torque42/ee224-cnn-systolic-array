`timescale 1ns/1ps
module tb_processing_element;
    reg clock=0, reset=0;
    reg signed [7:0] Ain=0, Bin=0;
    wire signed [7:0] A, B;
    wire signed [23:0] C;
    integer expected=0;
    integer i;
    always #5 clock=~clock;
    ProcessingElement dut (
        .clock(clock), .reset(reset), .Ain(Ain), .Bin(Bin),
        .A(A), .B(B), .C(C)
    );

    task automatic clear_pe;
        begin
            @(negedge clock);
            reset=1;
            Ain=0;
            Bin=0;
            expected=0;
            #1;
            if (A!==0||B!==0||C!==0)
                $fatal(1, "PE asynchronous reset failed");
            @(negedge clock);
            reset=0;
        end
    endtask

    // Drive on a falling edge and check after the next rising-edge update.
    task automatic step(input integer a, b);
        begin
            @(negedge clock);
            Ain=a;
            Bin=b;
            expected=expected+a*b;
            @(posedge clock);
            #1;
            if ($signed(A)!==a||$signed(B)!==b||$signed(C)!==expected)
                $fatal(1, "PE failure a=%0d b=%0d C=%0d expected=%0d", a, b, C, expected);
        end
    endtask

    initial begin
        if ($test$plusargs("VCD")) begin
            $dumpfile("tb_processing_element.vcd");
            $dumpvars(0, tb_processing_element);
        end
        clear_pe();
        step(3, 4);
        step(-5, 7);
        step(-128,-128);
        step(127,-128);
        step(0, 127);
        // Reset a nonzero sum, then exercise the longest CNN dot-product length.
        clear_pe();
        for (i=0; i<250; i=i+1) step(-128,-128);
        clear_pe();
        for (i=0; i<250; i=i+1) step(-128, 127);
        $display("PASS: PE forwarding, signed MAC, reset, 250-term accumulation");
        $finish;
    end
endmodule
