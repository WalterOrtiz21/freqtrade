// This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
// © LuxAlgo

//@version=5
indicator('Market Structure Targets Model [LuxAlgo]', shorttitle='LuxAlgo - Market Structure Targets Model', max_labels_count=500, max_lines_count=500, overlay=true)

//---------------------------------------------------------------------------------------------------------------------}
// Settings
//---------------------------------------------------------------------------------------------------------------------{
len        = input.int   ( 10      , title = 'Swings', minval = 3, maxval =  10)
opt        = input.string("Hold"   , 'type', options=['Switch', 'Hold'])
maxB       = input.int   (300      , "Maximum Target Duration")

//Market Structure Shift
bullMSS    = input.bool  (true     , title =  'Bullish ', group='Market Structure Shift (MSS)', inline = 'M1')
cMSSbl     = input.color (#089981, title =  ''        , group='Market Structure Shift (MSS)', inline = 'M1')
perc_bullS = input.int   (100      , title ='% Target'  , group='Market Structure Shift (MSS)', inline = 'M1', minval = 1) / 100
bearMSS    = input.bool  (true     , title =  'Bearish' , group='Market Structure Shift (MSS)', inline = 'M2')
cMSSbr     = input.color (#F23645, title =  ''        , group='Market Structure Shift (MSS)', inline = 'M2')
perc_bearS = input.int   (100      , title ='% Target'  , group='Market Structure Shift (MSS)', inline = 'M2', minval = 1) / 100

//Market Structure Break
bullMSB    = input.bool  (true     , title =  'Bullish ', group='Market Structure Break (MSB)', inline = 'B1')
cMSBbl     = input.color (#089981, title =  ''        , group='Market Structure Break (MSB)', inline = 'B1')
perc_bullB = input.int   (100      , title ='% Target'  , group='Market Structure Break (MSB)', inline = 'B1', minval = 1) / 100
bearMSB    = input.bool  (true     , title =  'Bearish' , group='Market Structure Break (MSB)', inline = 'B2')
cMSBbr     = input.color (#F23645, title =  ''        , group='Market Structure Break (MSB)', inline = 'B2')
perc_bearB = input.int   (100      , title ='% Target'  , group='Market Structure Break (MSB)', inline = 'B2', minval = 1) / 100

//---------------------------------------------------------------------------------------------------------------------}
//LuxAlgo Defined Types
//---------------------------------------------------------------------------------------------------------------------{
type ZZ 
    int   [] d
    int   [] x 
    float [] y 
    bool  [] b

type mss 
    int     dir
    line [] l_mssBl
    line [] l_mssBr
    line [] l_MSBBl
    line [] l_MSBBr
    label[] lbMssBl
    label[] lbMssBr
    label[] lbMSBBl
    label[] lbMSBBr

type target 
    line   ln 
    line  vLn
    bool active
    int  index

//---------------------------------------------------------------------------------------------------------------------}
//Constants and general variables
//---------------------------------------------------------------------------------------------------------------------{
INV     = color(na)
n       = bar_index 
hi      = high  
lo      = low 
maxSize = 50

//---------------------------------------------------------------------------------------------------------------------}
//Variables
//---------------------------------------------------------------------------------------------------------------------{
var  ZZ         aZZ       = 
 ZZ.new(
 array.new < int    >(maxSize,  0), 
 array.new < int    >(maxSize,  0), 
 array.new < float  >(maxSize, na),
 array.new < bool   >(maxSize, na))

var mss MSS = mss.new(
 0
 , array.new < line  >()
 , array.new < line  >() 
 , array.new < line  >()
 , array.new < line  >()
 , array.new < label >() 
 , array.new < label >()
 , array.new < label >()
 , array.new < label >()
 )

var targets_bl = array.new<target>() 
var targets_br = array.new<target>() 
var targMSB_bl = array.new<target>() 
var targMSB_br = array.new<target>() 

//---------------------------------------------------------------------------------------------------------------------}
//Methods
//---------------------------------------------------------------------------------------------------------------------{
method in_out(ZZ aZZ, int d, int x1, float y1, int x2, float y2, color col, bool b) =>
    aZZ.d.unshift(d), aZZ.x.unshift(x2), aZZ.y.unshift(y2), aZZ.b.unshift(b), aZZ.d.pop(), aZZ.x.pop(), aZZ.y.pop(), aZZ.b.pop()

method set_lab(string str, int i, color txcol) =>
    style = str == 'Bl' ? label.style_label_down : label.style_label_up
    label.new(math.round(math.avg(aZZ.x.get(i), n)), aZZ.y.get(i), text='MSB'
     , style=style, color=color(na), textcolor=txcol, size=size.tiny)

method set_lin(int i, color col) =>
    line.new(aZZ.x.get(i), aZZ.y.get(i), n, aZZ.y.get(i), color=col, style=line.style_dashed)

method target_bull(array<target>aTar, bool toggle, int index, color col, string type) =>
    if toggle 
        //find lowest point
        lo_prc = high
        lo_bix = 0
        x = aZZ.x.get(index)
        y = aZZ.y.get(index)
        for i = 0 to n - x
            if low[i] < lo_prc
                lo_prc := low[i]
                lo_bix := n - i
        hiT = y + (y - lo_prc) * (type == 'MSS' ? perc_bullS : perc_bullB)
        if hiT > close 
            if opt == "Switch" and aTar.size() > 0
                tar = aTar.last()
                if tar.active 
                    tar. ln.delete()                    
                    tar.vLn.delete()                    
                    tar.active := false
            aTar.push(
             target.new(
               line.new(n, hiT, n, hiT, color=col, style=line.style_dotted)
             , line.new(n,  y , n, hiT, color=col, style=line.style_dotted)
             , true
             , n)
             )

method target_bear(array<target>aTar, bool toggle, int index, color col, string type) =>
    if toggle 
        //find highest point
        hi_prc = low
        hi_bix = 0  
        x = aZZ.x.get(index)
        y = aZZ.y.get(index)
        for i = 0 to n - x
            if high[i] > hi_prc
                hi_prc := high[i]
                hi_bix := n - i
        loT = y - (hi_prc - y) * (type == 'MSS' ? perc_bearS : perc_bearB)
        if loT < close  
            if opt == "Switch" and aTar.size() > 0    
                tar = aTar.last()
                if tar.active 
                    tar. ln.delete()                    
                    tar.vLn.delete()
                    tar.active := false
            aTar.push(
             target.new(
               line.new(n, loT, n, loT, color=col, style=line.style_dotted)               
             , line.new(n, loT, n,  y , color=col, style=line.style_dotted)
             , true
             , n)
             )

method setTargBull(array<target>aTar, color col) => 
    if aTar.size() > 0 
        for tar in aTar 
            if tar.active 
                tar.ln.set_x2(n)
                if high > tar.ln.get_y2()
                    tar.active := false 
                if n - tar.index > maxB
                    tar. ln.delete()
                    tar.vLn.delete()
                    tar.active := false

method setTargBear(array<target>aTar, color col) => 
    if aTar.size() > 0 
        for tar in aTar 
            if tar.active 
                tar.ln.set_x2(n)
                if low  < tar.ln.get_y2()
                    tar.active := false 
                if n - tar.index > maxB
                    tar. ln.delete()                  
                    tar.vLn.delete()
                    tar.active := false

method draw(int left, color col) =>
    //
    max_bars_back(time, 1000)
    var int dir= na, var int x1= na, var float y1= na, var int x2= na, var float y2= na
    //
    sz       = aZZ.d.size( )
    x2      := n -1
    ph       = ta.pivothigh(hi, left, 1)
    pl       = ta.pivotlow (lo, left, 1)
    if not na(ph)   
        dir := aZZ.d.get (0) 
        x1  := aZZ.x.get (0) 
        y1  := aZZ.y.get (0) 
        y2  :=      nz(hi[1])
        //
        if dir <  1  // if previous point was a pl, add, and change direction ( 1)
            aZZ.in_out( 1, x1, y1, x2, y2, col, true)
        else
            if dir ==  1 and ph > y1 
                aZZ.x.set(0, x2), aZZ.y.set(0, y2)             
    //
    if not na(pl)
        dir := aZZ.d.get (0) 
        x1  := aZZ.x.get (0) 
        y1  := aZZ.y.get (0) 
        y2  :=      nz(lo[1])
        //
        if dir > -1  // if previous point was a ph, add, and change direction (-1)
            aZZ.in_out(-1, x1, y1, x2, y2, col, true)
        else
            if dir == -1 and pl < y1 
                aZZ.x.set(0, x2), aZZ.y.set(0, y2)
    //
    //Market Structure Shift
    //
    iH = aZZ.d.get(2) ==  1 ? 2 : 1
    iL = aZZ.d.get(2) == -1 ? 2 : 1
    //
    switch
        // MSS Bullish
        close > aZZ.y.get(iH) and aZZ.d.get(iH) ==  1 and MSS.dir <  1 =>
            MSS.dir :=  1 
            //      
            MSS.l_mssBl.unshift(line.new (
              aZZ.x.get(iH), aZZ.y.get(iH), n, aZZ.y.get(iH), color=cMSSbl))
            MSS.lbMssBl.unshift(label.new(
              math.round(math.avg(aZZ.x.get(iH), n)), aZZ.y.get(iH), text ='MSS'
              , style=label.style_label_down, size=size.tiny, color=color(na), textcolor=cMSSbl))
            targets_bl.target_bull(bullMSS, iH, cMSSbl, 'MSS')

        // MSS Bearish
        close < aZZ.y.get(iL) and aZZ.d.get(iL) == -1 and MSS.dir > -1 =>
            MSS.dir := -1 
            // 
            MSS.l_mssBr.unshift(line.new (
              aZZ.x.get(iL), aZZ.y.get(iL), n, aZZ.y.get(iL), color=cMSSbr))
            MSS.lbMssBr.unshift(label.new(
              math.round(math.avg(aZZ.x.get(iL), n)), aZZ.y.get(iL), text ='MSS'
              , style=label.style_label_up  , size=size.tiny, color=color(na), textcolor=cMSSbr))
            targets_br.target_bear(bearMSS, iL, cMSSbr, 'MSS')

        // MSB Bullish
        MSS.dir ==  1 and close > aZZ.y.get(iH) =>
            if MSS.l_MSBBl.size() > 0
                if aZZ.y.get(iH) != MSS.l_MSBBl.get(0).get_y2() and
                   aZZ.y.get(iH) != MSS.l_mssBl.get(0).get_y2()
                    MSS.l_MSBBl.unshift(iH.set_lin(cMSBbl)), MSS.lbMSBBl.unshift('Bl'.set_lab(iH, cMSBbl))
                    targMSB_bl.target_bull(bullMSB, iH, cMSBbl, 'MSB')
            else  
                if aZZ.y.get(iH) != MSS.l_mssBl.get(0).get_y2()                     
                    MSS.l_MSBBl.unshift(iH.set_lin(cMSBbl)), MSS.lbMSBBl.unshift('Bl'.set_lab(iH, cMSBbl))
                    targMSB_bl.target_bull(bullMSB, iH, cMSBbl, 'MSB')

        // MSB Bearish
        MSS.dir == -1 and close < aZZ.y.get(iL) =>
            if MSS.l_MSBBr.size() > 0
                if aZZ.y.get(iL) != MSS.l_MSBBr.get(0).get_y2() and
                   aZZ.y.get(iL) != MSS.l_mssBr.get(0).get_y2()
                    MSS.l_MSBBr.unshift(iL.set_lin(cMSBbr)), MSS.lbMSBBr.unshift('Br'.set_lab(iL, cMSBbr))
                    targMSB_br.target_bear(bearMSB, iL, cMSBbr, 'MSB')
            else
                if aZZ.y.get(iL) != MSS.l_mssBr.get(0).get_y2()                     
                    MSS.l_MSBBr.unshift(iL.set_lin(cMSBbr)), MSS.lbMSBBr.unshift('Br'.set_lab(iL, cMSBbr))
                    targMSB_br.target_bear(bearMSB, iL, cMSBbr, 'MSB')
                
//---------------------------------------------------------------------------------------------------------------------}
//Execution
//---------------------------------------------------------------------------------------------------------------------{
len.draw(color.yellow)             

targets_bl.setTargBull(cMSSbl)
targMSB_bl.setTargBull(cMSBbl)
targets_br.setTargBear(cMSSbr)
targMSB_br.setTargBear(cMSBbr)

if MSS.l_MSBBl.size() > 200
    MSS.l_MSBBl.pop().delete()
    MSS.lbMSBBl.pop().delete()

if MSS.l_MSBBr.size() > 200
    MSS.l_MSBBr.pop().delete()
    MSS.lbMSBBr.pop().delete()

//---------------------------------------------------------------------------------------------------------------------}