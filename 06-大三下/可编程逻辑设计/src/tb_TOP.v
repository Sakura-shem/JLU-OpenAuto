`timescale	1ns/1ns

module	tb_TOP();

reg 	sclk;
reg		rst_n;

reg		set_time_finish = 0;
reg 	[3:0]	set_sec_ge = 4'd0;
reg 	[2:0]	set_sec_shi = 3'd0; 
reg 	[3:0]	set_min_ge = 4'd0;  
reg 	[2:0]	set_min_shi = 3'd0;	
reg 	[3:0]	set_hour_ge = 4'd0;
reg 	[1:0]	set_hour_shi = 2'd0;


reg		clock_en = 1;
reg 	[3:0]	clock_min_ge = 4'd1;
reg 	[2:0]	clock_min_shi = 3'd0;
reg 	[3:0]	clock_hour_ge = 4'd0;
reg 	[1:0]	clock_hour_shi = 2'd0;


wire	clock_out;

wire	[3:0]	sec_ge;
wire	[2:0]	sec_shi;
wire	[3:0]	min_ge;
wire	[2:0]	min_shi;
wire	[3:0]	hour_ge;
wire	[1:0]	hour_shi;

wire	[7:0]	data_out;
wire	[7:0]	select;

initial	sclk = 1;
always	#1000 sclk = !sclk;

initial	begin
	rst_n = 0;
	#100
	rst_n = 1;
end


TOP						TOP_inst(
	.clk				(sclk),
	.rst_n				(rst_n),

	.set_time_finish	(set_time_finish),
	.set_sec_ge			(set_sec_ge),
	.set_sec_shi		(set_sec_shi), 
	.set_min_ge			(set_min_ge),  
	.set_min_shi		(set_min_shi),
	.set_hour_ge		(set_hour_ge),
	.set_hour_shi		(set_hour_shi),

	.clock_en			(clock_en),
	.clock_min_ge		(clock_min_ge),
	.clock_min_shi		(clock_min_shi),
	.clock_hour_ge		(clock_hour_ge),
	.clock_hour_shi		(clock_hour_shi),
	.clock_out			(clock_out),

	.sec_ge_r			(sec_ge),
	.sec_shi_r			(sec_shi),
	.min_ge_r			(min_ge),
	.min_shi_r			(min_shi),
	.hour_ge_r			(hour_ge),
	.hour_shi_r			(hour_shi),
	
	.data_out			(data_out),
    .select				(select)
);

endmodule
