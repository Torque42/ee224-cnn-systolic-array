module ProcessingElement #(
    parameter integer W = 8,
    parameter integer W_accum = 16
)(
    input wire clock, reset,
    input wire signed [W-1:0] Ain, Bin,
    output reg [W-1:0] A, B,
    output reg signed [W_accum-1:0] C
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
//W_accum will depend on the matrix dimension k  of Ain(4xk) and Bin(kx4).
//here we have considered for cases upto k=8, 24 due to the tetsbench
module systolic_array #(parameter integer W=8, parameter integer W_accum=24)
		(
	 input wire clock, reset,
	 input wire [4*W-1:0] Ain, Bin,
    output reg [4*W-1:0] A, B,
    output wire [16*W_accum-1:0] C
	);

	wire [W-1:0] a_ij [0:3][0:4];
	wire [W-1:0] b_ij [0:4][0:3];
	
	genvar i, j; //structural
    generate
        for (i = 0; i < 4; i = i + 1) begin : ai
            assign a_ij[i][0] = Ain[W*(i+1)-1 : W*i];
        end
        for (j = 0; j < 4; j = j + 1) begin : bj
            assign b_ij[0][j] = Bin[W*(j+1)-1 : W*j];
        end

        for (i = 0; i < 4; i = i + 1) begin : row
            for (j = 0; j < 4; j = j + 1) begin : col
					
					 wire [W_accum-1:0] c_each ;
                ProcessingElement #(.W(W), .W_accum(W_accum)) pe (
                    .clock   (clock),
                    .reset (reset),
                    .Ain   (a_ij[i][j]),
                    .Bin   (b_ij[i][j]),
                    .A     (a_ij[i][j+1]),   
                    .B     (b_ij[i+1][j]),   
                    .C     (c_each)
                );
                assign 	C[(i*4+j+1)*W_accum-1 : (i*4+j)*W_accum] = c_each;
            end
        end
    endgenerate
endmodule