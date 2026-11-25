`timescale 1ns/1ps
module ProcessingElement #(
    parameter integer W=8,
    parameter integer W_accum=24
)(
    input wire clock, reset,
    input wire signed [W-1:0] Ain, Bin,
    output reg signed [W-1:0] A, B,
    output reg signed [W_accum-1:0] C
);

    wire signed [2*W-1:0] product=Ain*Bin;
    always @(posedge clock, posedge reset)
    begin
        if (reset==1'b1)
        begin
            A<={W{1'b0}};
            B<={W{1'b0}};
            C<={W_accum{1'b0}};
        end

        else
        begin
            A<=Ain;
            B<=Bin;
            C<=C+product;
        end
    end
endmodule
