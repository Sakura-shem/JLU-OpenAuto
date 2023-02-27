module time_control(
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
	output	reg	clock_out,

	output	[3:0]	sec_ge_r,
	output	[3:0]	sec_shi_r,
	output	[3:0]	min_ge_r,
	output	[3:0]	min_shi_r,
	output	[3:0]	hour_ge_r,
	output	[3:0]	hour_shi_r
);

//=================时钟模块====================//
//---------1ms延时-------//
reg		[15:0]	cnt_1ms;	//1ms计数
reg 		flag_1ms;			//ms进位信号
always @(posedge clk or negedge rst_n) begin
	if(!rst_n)	begin
		cnt_1ms <= 0;
		flag_1ms <= 0;
	end
	else if(cnt_1ms == 16'd499)	begin
		cnt_1ms <= 0;
		flag_1ms <= 1;
	end
	else	begin
		cnt_1ms <= cnt_1ms + 1;
		flag_1ms <= 0;
	end
end
//--------1s延时--------//
reg		[15:0]	cnt_01s;		//1s计数
reg		flag_01s;			//s进位信号
always @(posedge clk or negedge rst_n) begin
	if(!rst_n)	begin
		cnt_01s <= 0;
		flag_01s <= 0;
	end
	else if(flag_1ms)	begin
		if(cnt_01s == 16'd999)	begin
			cnt_01s <= 0;
			flag_01s <= 1;
		end
		else	begin
			cnt_01s <= cnt_01s + 1;
			flag_01s <= 0;
		end
	end
	else	begin
		cnt_01s <= cnt_01s;
		flag_01s <= 0;
	end
end
//============================================//

//=================时钟模块====================//
//---------秒钟个位、十位--------//
reg		[3:0]	sec_ge;
reg		flag_sec_ge;		//秒钟个位进位信号
reg		[2:0]	sec_shi;
reg		flag_sec_shi;		//秒钟十位进位信号

always @(posedge clk or negedge rst_n) begin
	if(!rst_n)	begin
		sec_ge <= 0;
		flag_sec_ge <= 0;
	end
	else if(set_time_finish)	begin
		sec_ge <= set_sec_ge;
		flag_sec_ge <= 0;
	end
	else if(flag_01s)	begin
		if(sec_ge == 4'd9)	begin
			sec_ge <= 0;
			flag_sec_ge <= 1;
		end
		else	begin
			sec_ge <= sec_ge + 1;
			flag_sec_ge <= 0;
		end
	end
	else	begin
		sec_ge <= sec_ge;
		flag_sec_ge <= 0;
	end
end
always @(posedge clk or negedge rst_n) begin
	if(!rst_n)	begin
		sec_shi <= 0;
		flag_sec_shi <= 0;
	end
	else if(set_time_finish)	begin
		sec_shi <= set_sec_shi;
		flag_sec_shi <= 0;
	end		
	else if(flag_sec_ge)	begin
		if(sec_shi == 3'd5)begin
			sec_shi <= 0;
			flag_sec_shi <= 1;
		end
		else	begin
			sec_shi <= sec_shi + 1;
			flag_sec_shi <= 0;
		end
	end
	else	begin
		sec_shi <= sec_shi;
		flag_sec_shi <= 0;
	end
end

//---------分钟个位、十位--------//
reg		[3:0]	min_ge;
reg		flag_min_ge;		//分钟个位进位信号
reg		[2:0]	min_shi;
reg		flag_min_shi;		//分钟十位进位信号

always @(posedge clk or negedge rst_n) begin
	if(!rst_n)	begin
		min_ge <= 0;
		flag_min_ge <= 0;
	end
	else if(set_time_finish)	begin
		min_ge <= set_min_ge;
		flag_min_ge <= 0;
	end	
	else if(flag_sec_shi)	begin
		if(min_ge == 4'd9)	begin
			min_ge <= 0;
			flag_min_ge <= 1;
		end
		else	begin
			min_ge <= min_ge + 1;
			flag_min_ge <= 0;
		end
	end
	else	begin
		min_ge <= min_ge;
		flag_min_ge <= 0;
	end
end
always @(posedge clk or negedge rst_n) begin
	if(!rst_n)	begin
		min_shi <= 0;
		flag_min_shi <= 0;
	end
	else if(set_time_finish)	begin
		min_shi <= set_min_shi;
		flag_min_shi <= 0;
	end
	else if(flag_min_ge)	begin
		if(min_shi == 3'd5)	begin
			min_shi <= 0;
			flag_min_shi <= 1;
		end
		else	begin
			min_shi <= min_shi + 1;
			flag_min_shi <= 0;
		end
	end
	else	begin
		min_shi <= min_shi;
		flag_min_shi <= 0;
	end
end

//---------时钟个位、十位--------//
reg		[3:0]	hour_ge;
reg		flag_hour_ge;
reg		[1:0]	hour_shi;
reg		flag_hour_shi;

always @(posedge clk or negedge rst_n) begin
	if(!rst_n)	begin
		hour_ge <= 0;
		flag_hour_ge <= 0;
	end
	else if(set_time_finish)	begin
		hour_ge <= set_hour_ge;
		flag_hour_ge <= 0;
	end	
	else if(flag_min_shi)	begin
		if(hour_ge == 4'd9)	begin
			hour_ge <= 0;
			flag_hour_ge <= 1;
		end
		else if((hour_shi == 3'd2) && (hour_ge == 4'd3))	begin
			hour_ge <= 0;
			flag_hour_ge <= 1;
		end
		else	begin
			hour_ge <= hour_ge + 1;
			flag_hour_ge <= 0;
		end
	end
	else	begin
		hour_ge <= hour_ge;
		flag_hour_ge <= 0;
	end
end
always @(posedge clk or negedge rst_n) begin
	if(!rst_n)	begin
		hour_shi <= 0;
		flag_hour_shi <= 0;
	end
	else if(set_time_finish)	begin
		hour_shi <= set_hour_shi;
		flag_hour_shi <= 0;
	end		
	else if(flag_hour_ge)	begin
		if(hour_shi == 3'd2)	begin
			hour_shi <= 0;
			flag_hour_shi <= 1;
		end
		else	begin
			hour_shi <= hour_shi + 1;
			flag_hour_shi <= 0;
		end
	end
	else	begin
		hour_shi <= hour_shi;
		flag_hour_shi <= 0;
	end
end
//============================================//

assign	sec_ge_r = sec_ge;
assign	sec_shi_r = sec_shi;
assign	min_ge_r = min_ge;
assign	min_shi_r = min_shi;
assign	hour_ge_r = hour_ge;
assign	hour_shi_r = hour_shi;

//=================闹钟设置===================//
always @(posedge clk or negedge rst_n) begin
	if(!rst_n)	begin
		clock_out <= 1;
	end
	else if(!clock_en)	begin
		clock_out <= 1;
	end
	else if({hour_shi,hour_ge,min_shi,min_ge} == {clock_hour_shi,clock_hour_ge,clock_min_shi,clock_min_ge})	begin
		clock_out <= 0;
	end
	else	begin
		clock_out <= clock_out;
	end
end
//============================================//



endmodule


