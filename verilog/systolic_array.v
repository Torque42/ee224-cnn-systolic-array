`timescale 1ns/1ps

module systolic_array #(
    parameter integer W=8,
    parameter integer W_accum=24
)(
    input wire clock, reset,
    input wire [4*W-1:0] Ain, Bin,
    output wire [4*W-1:0] A, B,
    output wire [16*W_accum-1:0] C
);
    wire [W-1:0] a_ij [0:3][0:4];
    wire [W-1:0] b_ij [0:4][0:3];
    genvar i, j;
    generate
        for (i=0; i<4; i=i+1) begin : boundary_a
            assign a_ij[i][0]=Ain[W*i+:W];
            assign A[W*i+:W]=a_ij[i][4];
        end
        for (j=0; j<4; j=j+1) begin : boundary_b
            assign b_ij[0][j]=Bin[W*j+:W];
            assign B[W*j+:W]=b_ij[4][j];
        end
        for (i=0; i<4; i=i+1) begin : row
            for (j=0; j<4; j=j+1) begin : col
                ProcessingElement #(.W(W), .W_accum(W_accum)) pe (
                    .clock(clock), .reset(reset),
                    .Ain(a_ij[i][j]), .Bin(b_ij[i][j]),
                    .A(a_ij[i][j+1]), .B(b_ij[i+1][j]),
                    .C(C[(i*4+j)*W_accum+:W_accum])
                );
            end
        end
    endgenerate
endmodule
