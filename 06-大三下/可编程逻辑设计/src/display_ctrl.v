module display_ctrl(
    input   clk,
    input   rst_n,
    input   [3:0]   sec_ge,
    input   [3:0]   sec_shi,
    input   [3:0]   min_ge,
    input   [3:0]   min_shi,
    input   [3:0]   hour_ge,
    input   [3:0]   hour_shi,
    output  [7:0]   data_out,
    output  [7:0]   select
);

reg [7:0]  data;    //数码管段选信号
reg [7:0]  sel;     //数码管位选信号
reg [3:0]  display_data=0;

//============================数码管动态刷新============================//
reg [10:0] m=0;
 
always @ ( posedge clk or negedge rst_n)    begin
    if(!rst_n)  begin
        m <= 0;
    end
    else    begin
        m <= m+1;
    end
end  

//----------------数码管位选-------------------//
always@( posedge clk)   begin
    case(m[5:3])
        0: begin
            display_data<=4'b0000;
            sel<=8'b1111_1110;
        end
        1: begin
            display_data<=4'b0000;
            sel<=8'b1111_1101;
        end
        2: begin
            display_data<=hour_shi;
            sel<=8'b1111_1011;
        end
        3: begin
            display_data<=hour_ge;
            sel<=8'b1111_0111;
        end
        4: begin
            display_data<=min_shi;
            sel<=8'b1110_1111;
        end
        5: begin
            display_data<=min_ge;
            sel<=8'b1101_1111;
        end
        6: begin
            display_data<=sec_shi;
            sel<=8'b1011_1111;
        end
        7: begin
            display_data<=sec_ge;
            sel<=8'b0111_1111;
        end
        default:begin
            data<=8'b0;
            sel<=8'b0;
        end 
    endcase  
end

//---------------数码管段选-----------------//
always @(display_data)  begin
    case(display_data)                      //七段译码
        4'h0:data = 8'hc0;//显示0
        4'h1:data = 8'hf9;//显示1
        4'h2:data = 8'ha4;//显示2
        4'h3:data = 8'hb0;//显示3
        4'h4:data = 8'h99;//显示4
        4'h5:data = 8'h92;//显示5
        4'h6:data = 8'h82;//显示6
        4'h7:data = 8'hf8;//显示7
        4'h8:data = 8'h80;//显示8
        4'h9:data = 8'h90;//显示9
        default data = data;
    endcase
end
//======================================================================//

assign  select = sel;
assign  data_out = data;

endmodule