// This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
// © LuxAlgo

//@version=5
indicator("FVG Trailing Stop [LuxAlgo]", shorttitle = "LuxAlgo - FVG Trailing Stop", overlay = true)

//---------------------------------------------------------------------------------------------------------------------}
//User Inputs
//---------------------------------------------------------------------------------------------------------------------{

fvgLen = input.int(5, minval = 1,  title = "Unmitigated FVG Lookback")
smoothLen = input.int(9, minval = 1, title = "Smoothing Length")

ts2Tog = input.bool(false, title = "Reset on Cross")

bullColor =  input.color(#089981, title = " Bullish Color", inline = "bull", group = "Style")
bearColor = input.color(#f23645, title = "Bearish Color", inline = "bear", group = "Style")
bullFillColor = input.color(color.new(#089981, 80), title = "", inline = "bull", group = "Style")
bearFillColor = input.color(color.new(#f23645, 80), title = "", inline = "bear", group = "Style")

//---------------------------------------------------------------------------------------------------------------------}
//Functions
//---------------------------------------------------------------------------------------------------------------------{

fz(_val,_standin) => _val > 0 ? _val : _standin


//---------------------------------------------------------------------------------------------------------------------}
//FVG Level Arrays
//---------------------------------------------------------------------------------------------------------------------{

var bull_lvls = array.new<float>(0)
var bear_lvls = array.new<float>(0)

//---------------------------------------------------------------------------------------------------------------------}
//Calculations
//---------------------------------------------------------------------------------------------------------------------{

//FVG Detection
if low > high[2] and close[1] > high[2]
    bull_lvls.push(high[2])


if high < low[2] and close[1] < low[2]
    bear_lvls.push(low[2])

//Array Size Management
if bull_lvls.size() > fvgLen
    bull_lvls.shift()

if bear_lvls.size() > fvgLen
    bear_lvls.shift()

//FVG Mitigation Detection
if bull_lvls.size() > 0
    for i = bull_lvls.size()-1 to 0
        if close < bull_lvls.get(i)
            bull_lvls.remove(i)


if bear_lvls.size() > 0
    for i = bear_lvls.size()-1 to 0
        if close > bear_lvls.get(i)
            bear_lvls.remove(i)

//Bars to calc SMAs
bull_bs = fz(ta.barssince(not na(bull_lvls.avg())),1)
bear_bs = fz(ta.barssince(not na(bear_lvls.avg())),1)

//Progressive SMAs 
csum = ta.cum(close)
//Calc starts at 1 when FVG array is empty and Maxes out at Smoothing Length
bull_sma = (csum - csum[math.min(bull_bs,smoothLen)]) / math.min(bull_bs,smoothLen)
bear_sma = (csum - csum[math.min(bear_bs,smoothLen)]) / math.min(bear_bs,smoothLen)

//Displayed Extreme Values
bull_disp = ta.sma(nz(bull_lvls.avg(),bull_sma),smoothLen)
bear_disp = ta.sma(nz(bear_lvls.avg(),bear_sma),smoothLen)

var int dir = 0
if close > bear_disp
    dir := -1
if close < bull_disp
    dir := 1

//Actual TS
var int os = na
var float ts = na
float sr_crossings = na

os := close > bear_disp ? 1 : close < bull_disp ? -1 : os
ts := os > os[1] ? bull_disp : os < os[1] ? bear_disp : os == 1 ? math.max(bull_disp, ts) : math.min(bear_disp, ts)

if ts2Tog
    if os == 1
        if close < ts
            sr_crossings := ts, ts := na
        else if close > bear_disp and na(ts)
            ts := bull_disp
    else if os == -1
        if close > ts
            sr_crossings := ts, ts := na
        else if close < bull_disp and na(ts)
            ts := bear_disp

//---------------------------------------------------------------------------------------------------------------------}
//Display
//---------------------------------------------------------------------------------------------------------------------{
ts_color = os != os[1] ? na : os == 1 ? bullColor : bearColor

plot_upper = plot((not na(ts) or not na(ts[1])) and os == -1 ? bear_disp : na, 'Upper'
  , color.new(bearColor, 50)
  , style = plot.style_linebr)

plot_lower = plot((not na(ts) or not na(ts[1])) and os == 1 ? bull_disp : na, 'Lower'
  , color.new(bullColor, 50)
  , style = plot.style_linebr)

plot_upper_ts = plot((not na(ts) or not na(ts[1])) and os == -1 ? nz(ts,ts[1]) : na, 'Upper Trailing Stop'
  , color = bearColor
  , style = plot.style_linebr)

plot_lower_ts = plot((not na(ts) or not na(ts[1])) and os == 1 ? nz(ts,ts[1]) : na, 'Lower Trailing Stop'
  , color = bullColor
  , style = plot.style_linebr)

plot(os != os[1] ? ts : na, 'Trend Changes'
  , color = os == 1 ? bullColor : bearColor
  , style = plot.style_circles
  , linewidth = 1)

plot(sr_crossings, 'Crossings'
  , color = ts_color
  , style = plot.style_circles 
  , linewidth = 3)

fill(plot_lower_ts, plot_lower
  , (not na(ts) or not na(ts[1])) and os == 1 ? nz(ts,ts[1]) : na
  , bull_disp
  , bullFillColor
  , color.new(chart.bg_color, 100))
  
fill(plot_upper, plot_upper_ts
  , bear_disp
  , (not na(ts) or not na(ts[1])) and os == -1 ? nz(ts,ts[1]) : na
  , color.new(chart.bg_color, 100)
  , bearFillColor)

//---------------------------------------------------------------------------------------------------------------------}