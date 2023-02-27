module TOP(
	input	clk,
	input	rst_n,

	input	set_time_finish,		//设置时间
	input 	[3:0]	set_sec_ge,
	input 	[3:0]	set_sec_shi, 
	input 	[3:0]	set_min_ge,  
	input 	[3:0]	set_min_shi,	
	input 	[3:0]	set_hour_ge,
	input 	[3:0]	set_hour_shi,
	
	input	clock_en,				//闹钟开关，置“1”打开闹钟；置“0”关闭闹钟。
	input 	[3:0]	clock_min_ge,
	input 	[3:0]	clock_min_shi,
	input 	[3:0]	clock_hour_ge,
	input 	[3:0]	clock_hour_shi,
	output	clock_out,

	output	[3:0]	sec_ge_r,
	output	[3:0]	sec_shi_r,
	output	[3:0]	min_ge_r,
	output	[3:0]	min_shi_r,
	output	[3:0]	hour_ge_r,
	output	[3:0]	hour_shi_r,

    output  [7:0]   data_out,		//数码管输出
    output  [7:0]   select
);

wire 	[3:0]	sec_ge_rr;
wire	[3:0]	sec_shi_rr;
wire	[3:0]	min_ge_rr;
wire	[3:0]	min_shi_rr;
wire	[3:0]	hour_ge_rr;
wire	[3:0]	hour_shi_rr;

assign	sec_ge_r = sec_ge_rr;
assign	sec_shi_r = sec_shi_rr;
assign	min_ge_r = min_ge_rr;
assign	min_shi_r = min_shi_rr;
assign	hour_ge_r = hour_ge_rr;
assign	hour_shi_r = hour_shi_rr;

//------------------------------//
time_control			time_control_inst(
	.clk				(clk),
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

	.sec_ge_r			(sec_ge_rr),
	.sec_shi_r			(sec_shi_rr),
	.min_ge_r			(min_ge_rr),
	.min_shi_r			(min_shi_rr),
	.hour_ge_r			(hour_ge_rr),
	.hour_shi_r			(hour_shi_rr)
);
//-----------------------------//
display_ctrl		display_ctrl_inst(
    .clk			(clk),
    .rst_n			(rst_n),
    .sec_ge			(sec_ge_rr),
    .sec_shi		(sec_shi_rr),
    .min_ge			(min_ge_rr),
    .min_shi		(min_shi_rr),
    .hour_ge		(hour_ge_rr),
    .hour_shi		(hour_shi_rr),
    .data_out		(data_out),
    .select			(select)
);






endmodule