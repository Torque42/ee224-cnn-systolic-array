module ProcessingElement #(
    parameter integer W = 8,
    parameter integer W_accum = 16
)(
    input wire clock, reset,
    input wire [W-1:0] Ain, Bin,
    output reg [W-1:0] A, B,
    output reg [W_accum-1:0] C
);
    always @(posedge clock, posedge reset)
    begin
        if (reset == 1'b1)
        begin
            A <= {W{1'b0}};
            B <= {W{1'b0}};
            C <= {W_accum{1'b0}};
        end
        
        else
        begin
            A <= Ain;
            B <= Bin;
            C <= C + (Ain * Bin);
        end
    end
endmodule