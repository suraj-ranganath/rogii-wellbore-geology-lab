"""Exact medali CNN-MTP inference plus guarded override."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import zlib
from pathlib import Path

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
RUNTIME_PAYLOADS = {'inference.py': 'c-rM%-E!kLmcG|hV3^tpIgx0qlk_C2QK}qU?smnoUADWsrZg%REkY986v-hd+ZvCza<vbzGgo`H&z2|Ia}EFkq$s<4{?b{yTy2TO&%wdL`S}ijb{wY@r!!vgbjtmF-5FXxo%7RIM@Q@(Pm-&w;OrI8l5D<aX9bVqX&GlJ`}wE8v5SL4cBI>27t<on%TDJaiK953v)ee0vRfv~A}slQJz(X6v)r`s=Pczc7Hpc`KnoGOS~L8Vgn16tJF>c9*^KpHLZKy86#~i$K40>*6wLj)|ANg*#>45t^Vtags7^l@?3<n{*IC4(%!)O36&2xa%4S8j)WqDx*-DsNhCI<_BBN5vEaFKgTtOpzjx@wKJXtds*7RCXiK}dtMj><&PK!*)XR}q3umnb4go)qj9EYnE$fK@9_Cq+IC!D2O$*;2PnoWVxgARkA-{k9Z0nDjQjRSMl`HcPix1ZpborUFs%~xR&sVBd>-%h;Dv!Y~TE%e7sbb@!M=RX{syb2DE&e@PTJJ-aeo!hLq27v%^ov~UOU?1=aTk&(?iDaPu=EtevMd|K#Yn42Ukr!o%(-yHLzAA9~epSqyPxYs1xz4%hkZj@5VK_{^LRsl3c`tZK>r%bCqzCflG)odL#a_RT_$*u{C9Ve5z@MU}eu<wgMqV!BBI@N~QLfpqSsv5XGKWPhSeolcIf{tDXL+Qbfg+eL>;+k5?}2-$C{_s<zZw?Ec@@RfpJwSSt|Vjs@Ku+czc~rsj?OQRPESm^C@e$4OI^^V6_;dT6x@Oa3-EQ9U9I9I3PiZf6W*l~7hU!`_<nSFdOixy=*!;W<>*{}zC69$JMOYe^WpX0@nN^)nSMzH2O^s3p1~kpl$;kQyfhQ>|Na+7x_^3dc=YPc`Q9Z?%<~y}XaDtR|A({Fqm#>~eTdV1Rqkk8Dt4AU3X{11?YA#>ifkUoy=j{EmSx@piyfys(*>Vi=UEIM#E#SXVei$e;}JE9KSLCL|6uQOFE~Fvz3e<M;rem(<LUVy*#7Zo@8r$dbCNEwc5IsPFb$S`3C^M$ah?yrvC9&WiOXWeJ=Xh<oq)?K-HbUZO){FUM-RX`p*r*hf68KQzTl5E)BgU!@v-ZGjh^}<^ItlizYLQkn+9bOFWp_Qs>AQ5JTKXZ{(xz&x0Z)O&;znY{&W?EK40eLI+%hV%z>)sCA$}wqcheYu*0L117?)P#dDM58H6%{rQ=-lIf&w7fbm$DL7XK}r0^R_b^<|?6fg`n@b4isB%>QHz{jjwVL30@EK4Hzr=S9J2#8YKVSso?wvg~NTyiY93`^)Nnguuas?bA#joiaup5Koif?_H7{rN+1U#!vxKL@`R+jz&>H7qNJv0@ty7=|JZ`2IGAzzDx2*)7t16DHIuH3YE~I1v#B3>2Oz+7zkhyj&IF{D#B`j5#UF%C69{6A%d&rHoLiQ{W80^K0A!!ic|ULa=Is?Q&JaIuEd*fI_lKSJ(6BNp|JV9LXXzJ(5MgMP@jj*U@?gD0i!-JT@G%U85Vo#tBZWKgr5N3`|Ewkri%DdCZ)XjG}i-P+(|Km?qT_fb6=j59PuCW9l?o!|o_}77%a`*gZ0_F|^NV7@vFIgQ={B^cj0qWUE}Ww<~8{O(ITUHQiJkKLpD#2PRrUii>@EszAYVq`YF1VGjl=LYey97bRc%B2VJdbppqzUg`piLaCJBm9Be7t=?g)vI&@nJE)JJQSMv^>%c5^$w-){kp7Qt3hT5q1?3E3v7omMnlfFehYh=vrDdG1cztwnbO#i{oVrz;e!=r3oN_gmE(o{R7zivWs5JwZzjUi^O+6yKQBP0TSa&<|!#w9{<hr`aF3Yueqp}HLx=He(tMR0o(u!=1`Q_@-ZTr`JEnLqt1E+Sd{jO>fm-CP<<J8?G2qkVkOv2?=6tcSkyBqILyn0ibBu#tOs415h#7##InceH=A7DXL#eGyd51=<{Q-o{C)#_?^UlBj>znVRGj^a&0N~%uJGc@uq2JHJcN5=>3^kl@2P7X)sqm%s+dk2R9^xbna_cGY7(853>nQj%1OX(UgRFU0=%+-E!5r{=RLo+99fG|H^q2N(a-jqRD6yX{wq`8l98hm-Q9_ZZM`br(a%Df;q1j($sA>ZQwnY-X&32U~?6q@UiN<alD>Tq+Oi5R_QW!R4*ut@O~)K;0*kO}Qxqccz;#LIB5An6nwqtz6+A=M8-YxMyau$nnKKw-+}1=&uh0$OH4h2XPxE;cwN7Gchri*|ew%nPsUNtOELi;8iVZ!gh*S#cJYjDtB|vwm+@YDC!@rmpclpCMF_<H5e!*#%=i|JQ$`r<1^>(hv-3%(ZFuOf!2R0^r3w+DrpCw^4J`(3zr85Kg=1r7i}oV1p<wz&dvsifee3t;p@t)7$XNu%7hqP}2`sQykz!fDLt7DgQnufiSv2@Hz4V>*2?FK^5HU1xkU`x0rFrSdra|pv>si6vze%fk`RxIl6=K6mdTo70jDglV6-*@;0g^Id<?_AeLDMex0;$WlJf1R=vi4o3G&+yr%-ou$aeba3y76Xv1ynx3uK!VxgSlp#{V-T}*jo7vYSTYv~_qztd6?zi;%29+!FhMZKf%SS9IAKlLq&c&meInj7Am&ktHoDCHtp(-C!34*iesYt%h-!6VfauwIS0hl$tXk1AkyzsPbH!XnYxr)v5Al6s$JV;B8b!|`5yK4h7Gy-a;plyFMnhl0qQalAW-`xEfzNJF_5Hj9$s`AZ%`;Bgm=Vc$b8xi7xzHd%JL&N&Jxux+{ksR&WS##V+6L=AP@zlJTJu{VM@^(~NKaRI@6Ntz5Dfj0OJliP4Dl#)|TxjZHfUsLWV=q82PrVZuthaPds4jaUb4J8x3tUwXQ0#u<Z>sV{#Gj<3e%vCtOM(aiD7<q?F(a<2WC0~FRTE;?P-qbSal7gDz(fZ$+)JRcUCKsfH(2oVG!YkoQ&C^q@c|J%G`q0Y1f4|#h?@`RCYEMJ=^6MAsYZbS?-vzy~i!6jXQYn;mrI3eFfC0ubyu-v@@4W?w;k&}TgC=^W2~(`pB8~A@?n5mqF_zV!!2!}=RqVd+dy_}4g-b+rW&aS9VJ>3m@MzARX0MXpRj=^1wOeH5KcQcq2rJUF$4vu%zdLx}pR`gaNmB*t>-V)ZtiL`dTIw@J{ZLzS;KM+M(Gw6Ct?tJ7pvwjmkY|CYEK0lbz?ur}b<H<ya|Higg9B{<O9wYQR1WJjOIAzpp>|O}ST?`c2Z2E(klij~C|@eAtlB}RWttVB=JO1#Xqnl2GeAkAmLbg@Zu2-lV@EToj08IgPnf&m<N*cZI?iPyl2BPyXRU9fx2wlO3+K<g$b{=wxj<!spsfhUFx;3YtPLJt364r6o4}11ZY|xYwbDitI1B|L4cHU{1<>}un+kCNsu64`!gS8<CTl91+;Tz$CCj*O3A_|CFPmFe3^r0~9A+2PC3YvhF!(}$c6OKv2~grB{?f21vXzOA6IC`vRq~#Igg%pS?=;B-f^J$~NK!~TU>sFQVthDw+prc|$@Db2O_R}_v@wi&0<9q5m1G?h;cX=rNsJ&@nye)<8A7y&<L`kVA%#jjIbA?Z-(mm=o{z28Jx(1+)E>IfWLNSO2Dw4D=WP~;sq1%Ji{qP(D)2ljhJ7C_8Zdc!or0hYp#dZieW@W*dwg^<3ikH8TOHN6$d(xVb&P#J?F|UZ??gi7dIeQ1jHkJ|%#}pP<UE*Jlue8WO8TZePTU?8+L6wxhcdP6?^f0O?O}*&M)ePpVz=kU+R30=Ue)+Ox32uA#%L~|9L@X2Xi7C2@OT=8Z*!GYjmwH~x8V`hX;j$ZeO2F4w$dRCM#t0EYq=iO>*nHcB;Oo~;4B(3w^Q10%gwh+OE~nl3dK>NF@BSo7AxYmyOkE)k=@EKUAY!kf`jfSO%U;`)qJkK;fH4kl72bBts-wn=a=mJ(>Esvd*^>(=cn&3G)TeT?j64wJ-h6vB~&h1QGCFj056SIs}&@`(tNGMmir1HGCozcNwftrl6F<tc%de&QjHz~rHsN>Wm5lgad7w-kFsOhOOrOSvq5&HlN8h_qhD?2kp!8|wL|r3ktsarQg4IR;}H$Vk_P4O{`}KlUw-M;_r!wN2=R4US!r@$HMz<m@-fdVz%c|fpWuerRZfl6P*8LvJKF6~Y1||uXjfB#6l>%V`4<-8jU{oO1OLH|8Shrx%zFI^K><uipiAzj4d-E;g8vhyYZ+oew8f?&_(U*X4vk61D{YEM0^(7<Y}O;ELnnr9=(0A0$_x<macou?Z%?A_CTf8q$<<>aQ-fcO5bH<=gao{0Gf;AVY!gt4#n^6g(i#apKfU?YfL$CP9UPs!QVAt?c0M{d+Q;p74E#w-I=vemA3rzEsnRSe!4xRTwGvGqkYkcMM37g2>H5bDNBjnKBo)RzE030~l1lY-(l+g^&M{bePi@$59#4Q6j&_!ya#<m~#zMKvy#0)WZ`5V8C7It>U?hZVU-SuCE>>w3e{=Yr&7|&L$WEnVgwAAQM|B4z-=ew#FUdmi^|=U{$&hCwJW3y-#zhV_&q!D>g4C53{#k=IK`qz8fttNivpT5u&RDy18S2w8MO)`08}v>NE91lz%VHZ^mGp#QL&G_jyHa>6`0^^@*4c=Lrrk)iJ(*(DSuI4W6N7OGGC&#4P@GPa6-_o798i-TI|V9vQrR(+^-`G;IolL*gKbe@vb9NYRWAHSuB^ZblTI)gB0cH4g_H2aZ_6+}nPXZ=Ux00ggf()F1UHJwr%nYI9;HAqM4cOO`#=>?F*<oTH3S{lu8U-%#4=cfFeEh_V3%*T)GJZ<+7hV-?lIG>(u&(cuLg25P=m$DoN^#&7VU>0nq8yuy!iShEV&O%v{4PYNHtFn<3f~rx?`QEK!#|gp_6kqYqn?sx$4V!$vcho{!lkYFTjt3C51a&ZBT0;^L{uB3s{1%^0L)@0WJurgoc}FgH9ir0hB!*!-ArHP#{fu?BwFA{E=X@uwaw}!9y6C5HaY0&X%nUL$)o^SF7=8GIOwDX|7;JUL&z&2Vzv%850XcF-bAu_D8t^kyv8!QQ<Rk6N4Vzr-j9|?p0Q#4!7+u*<g`%{=;y#+gNWhkvOc<(FV?H?0Gh6;W}9~5iR@$d!D1cxI(mzm0&5d;@=K~8B9rnDFLNgXm>Um(|}GFxh-rSS6IwMWhPX90#g(+71{6hyDz`$zWB<p%9EPS_53n(rLW|5It@7UV`xx-&p(d_nvGzduYiX8YK#NJpRR$K28QB?mI)qd+i$s5uJR7_sY`S#gvb8wX)BalE9_LnBfw<Ga1i$$6O+>nR8JVW4K;^aLdMTvz@@visbz5MFZPf2g37oR1HCbB40a~A;%u7_YbbJ)Yc>0b&bNBdx>wAT?W+0SC|H5`QM?@Xz47kEpXMu=b}3I@y|msL&_b1B@E3&li>=wWdg&dh)g|A&mI89Z$jNZ4r=A@Vw}{lf*}-FQm|tEKgRay<XoFSNu)5WPH8AR=ZG7*Fxa2jD8W@rNlarTLx2Kq+xus6gQMOzXYt5>Z4P{CM`{w7L{`%re&uZj#EwL#ir1U#a1_KEmSY6veO6Af~bXDK((v(RnPX{3i=3~C;*z2>AJ<`-SS900bmz}$77RRWL>8;1Qw#@H!t=vMxEjqzHe{>$+u=G03Zd2`8V^Fz`GTd~azjm2%XT7EqRMq)zs7b|**4EQ4ew#7I+qZ;{bvpyuG<w*Tc#k)lvz=<<XRXAfMZ))@8;mSJCBLecPz7HRrww=go0wbqwK~$&VPZ&Qfhg0rn%M48$MTW^)tH7bte((^wpxlywp@Xujp5}Lr)W#nF$ga2$;35=x-_@eW=?E5P`y31sZ6J!tt!9Y?*HS4Ege|va9lIzbD4Aw?ZG}Ka8{RfVXMoQrJ8zGjHe>2GRu+pTmCu3J^E&f3LdKVP;3Sf+t-Tav?>U;PWCfIS-&2zi=&fQ$D`h>vp3creitFOg!iNjLolCdq|JMFnW|N4g<V-1NcUO-u5lz3+W9n$Dvze&*%R@B6)NEM!?0J7YOzX65zyxS4?uw#ow#CF2Y8&j<f3#Mr))LEbzzGIIr_a#G!Q~?io!vp6fk(28k7Xj%M$u1cc+%P5fyO;BVtN(`W1xRf2r?~F>tEo%OC~<wi%*LjbzEHBiyZDE=Y~?bp|$7-BT0m4W`yfMM6I+7?EJr3PhG$Qz<P5bR!lygW~$kL7nO@y-#>4*h(V6v{fbq*Wq@-)ky`JP?KBNg&G3w1Dlq{4n;a`j{BApeuvvi5PqqHj+GHU&ccYWQ#$mdpx{po;xH$(#`jXVA0|XN@{hKLn>FG+$dOnlBYodjTQ1e`YRA&5s^~qMcC7tI@T`i%@UjfkTbE%Rj7dzCsG+0{JbWI{7qFmk+tctcOoVj^Fy(X*z=GK8aotrYcraZIu@8R^Z=LCE?sVDxgD1aU92^GU?_G?7{nzjquRE}w*rB}@(p#O=2+*%9IW2e^%-B&kbl_uWQm?>45?%91@=%|ng92?Q=&X5?7mV;|!0s!)$y0Pra~sqVrcX<mIr-$TVd)rE8z>EmIEg|(P_mS&r(E}+qGhG}yHHI`hcm};$-z(}yx~K-s^F1J2yO{ps9)Kl8}3TQJfwfU2E<DFfXGS&*soKHaBv}Eky`4(3Qe1tgMp4^DA!~s3A$BoV2HlzBR-#%1W2C(|F?LJ1KO)j2>?mGwMSmRYYsx+AVqemZex^b`4`6*e;-JTh^*Lx%3d@e+4}Ct4~8_Kf;H=c(+VKAY|H%Ej$5$IkL>s#2QQ_hXykIBlbN)ArIzTl;LA=sUiM*tWStXFVUiz=ODdG`17OL=LCFf3d=isXJrX22X)W=AfaEqL^5fCSmicJljcmt8DYu}HD!bMOJ<9C`EgPC@mH%OUumvORzpT`4l|B3ztZ)PB{xB%51+*e<E1|Rtxsh7xU$3p=7GUiH%)n;(2H2Ihi}@y`^$){ZPeWS`c&j!+1R@oD86t35G~x+b<8}l=_i}EV7%Ui2P@v8nfC$s3Hupcx3AEZsJzQ5!$Pi_`-jv6_v$rxg``bLPP_I7j^(T|3c)pE7WWH{$&ceE9HC8UpWP_T{*87$acWNYqTOHJ&|L1=hyH_1R^bp1gd8_20*4A#*8Pg6K)47_D_REw?FfBAa7=N0fK5U;I2`H2Jj&0v=($+_<1TM-!#e8a~(+(rF&49h7dn@N(3|fTZL)foQOqzLk(VU3KsY7Q$!*vGO7|RM1x^>2`s{Hsw*-sfdH?I<te);xb^Z<bigd->#t#<$ssSIc;oB+eo=XEjN-qDE#6P_KNjqps&bHW7i^4kEz-XOpmpPV2-qYwnnpks-ZGjzaqzWll=YFqAe-n_*5=H9x;ORt+|+x4is@2l-X;SDUg5HYz|XKy}%_?yT~(NPs95l$E7sRcF#4^H3TCm#{EGjp)e;68qE$eoFeGb39Avx5l^*)kYEI*QB24!EFt6P@DxA0aZ{tFr|(lB}t@O9xi43q``f&M-vmH!#`Qa9tgm>zg8dfz`nA#{G%6#bAMs1ntP~EiTM?sZeDSEEjG0CaChlS7!sQtHM4#AjM|5M)qo#n<2z-U%y23VW4KfLdb`k7wFY1)o#XZF0-exM0v}azrp1%6_*hO|3vXCxiJ|ppQ3y2kX6pTxj*$`hY);IN+v0)o9={-RL;zuVL`CSiuljCyEP%-$DfKDd5KspKR?4Ik_jIZ7(jrlDXHI&=7UGBZc%G*SQhuZ4XL@TI6si{s0Y!qCcg3|?k!eVbnODTEG1v&cz%QZa&sduHVS|!^l~E~w@U;tX16g<rko<)VvLdH3o|whG!;`!U2o8}7;_dDf>($o?QDmLRBpGB*EpNz;>bRxM913cdPt}8;)46Cwz@8mbX=A9GXAciXY9lg_`9Mk{Gz(j4`M+)P0R1_9&gJDv#Y629dzxgZgk{h7bKd<`?U-9C!3*uB`9C}?1GYAa!D(w=NC1PiokQ!L%kxqf>m|3Ry$j1x<tmVGKegRO0W@_fdT_oFuaH#BNXZ53jQZS*U+hS;gF#_J7D99I)Zs)*|dks(!~f+kh$aOYDsv#S!u?`HCk8Rw;0FlKbUcm!4g-u2|}x$!OrX54_Z<TNbQ4`>doFJ+U?$0{4afVcRt-FsvW0K&lr+cJ71+KUHV_0nW&OkwWNbQbb_N&>8&R@iI|P#Lmk8`!vck7xGjTS&kaLiX(d|cplg}tZ8>Eq*VSX`>NK)_RK6li4o}OA6-k%!_D$E4pHiMMq#D+k{r4GOWzcD|d5`w`NcPFRX~{Q@{O^4BBQFpFR;-u*9SGF@w`&BM&O~A&BLI(0HyBq%HKh^ZeFM&ujrpq_lg_S=t7#>58m8qkE5oE2)T*T@n`yDR-}q`9y=6)EFW$0GZpjgqxaKSa>sP{ndpq!m>nhi{t;I*jagNpT*ZO&@>axR{QTBB<ULyXHvWW$vzm-+1?UyxS0~M+)npq2Bp$`+O3B}na-D*h!7P2p(RV}MMQ1G@P8*1N^OwUzsBsd7A$EpE~R#`#jXz&&h3b?Y;Tx>!QDeh9OZkTIAzH2T5F8yTf8uoXYy3WiT!J(_m)(UgQS;W0^gtUTg>XDg5%&RA{GkusTMYu;^0<l2?5o~UW*F}A?TDlGF)GQH+YSqg=5G{7<2;+C$QX=<(0F9g1$`U?Y5nJtmX*k!nh%(L#x;5DvfbM_dV|F&bTDz58HD(wkV!O!H#Rlhb^P1ufR1yWlK+`Bx6U2(yA0OXbByYcgIdtolm9*%qH7e|}c8gXkA<Zh5Kqc993~cksb3*H*WX^&`>sstva0tUR)>kWjs=E}-VL5ryvsJF;tBpX)`cJ}eKL~_t@l*^p;tSk0KnXCV_^sYlXwfW>_YcyJv3skmyp_-!_%CK{bs$z`n|0vn_9}S5zonb_kzSVJU0ap#ZhIA6%Cbs5x1l(oR=l<vp}_yliak=4u7L*$#CFx3NgYN@x-uJ%-P{ICKpW9jMAeUTW@#Hb3z+h-=wT`vdMRSuf9*8!z+GP@7A<S_iHSB)rmqBV{y$?ojkvgJS18WZOMT*(8HL!o!LAIV>|I;-+Nifpx!-$TbyG^tABnrS;<DRv5B~?m^a+>', 'src/__init__.py': 'c-m7|NGvVM*G*5&OD#$)Nlj5ms#GXWEh#NfNKVbk;Q|1zbqen', 'src/config.py': 'c-rL|U2oeq@ZG<H&|c)tmeVxtLjVs|WI55`ScW8T?XrN7CCcF}ivlU7&5Q#55&ebzB|B2GWINu5y)Cm~2<Gwb$m4zR?xbm2NAPG6u`HoWBY#5Z+n28h`J*w)X--m_k<RcRo{mMbf`=)pxRg9BD9cd7I4T|px+cso2;oyw?C~`^8m!)->zwfdF|qx}!z^a&PQuw5<zexV(gmugW-x3hvT_ak4F$^bEvR5Tde|EnF%<=;i?SfG#FfKNr}G{9_Vvpt{6JFs{mk~9TYDNHX9RC2_;5n<kIcZtKii(~xKlJl+AFQ&&8H2d2F*P8C)*0})V#GNh_fZ7{f{I~7mSnslCYF5pZX$Bv=g%)d49-q-2jao-*VyHPRV*YF-cHDvlz!=5n|3*p?6XLM-_=KI|!Z#!WD&CBEWF&6i7Z2LDvSWaJfv$fM$7FK$KnQq@Xf7iosu)Sk6gIBZ;DQdw<LAUo$YGkFmZ3`@RJldB`D%i)!*-y=DBC0wbF32zl9%(*)^iet<Lx{}k3UsK7Dhdbh=r4<|**GsVo{M6mRc=v{>YWP+eEP05rMV^(Id%{k+GL#|n@Ok<S6GExm_=LnHhiGc!6NqqTRrZqG_APt7b?~G=8q9L@2K5I&vXkV0S3tez%tfGXn^5HXepP0j2a7B3L1X=EM>|7&jV*36md3=<HLLh5=UD@zfax?;q_@A7D609|OuzARl^wtR~#{esErg-Fd&k=T9zO#QyZdpKVLjQ{sx^+kPWP1Qj@cqd6ie}3}^gyCj4l$+>gDUp8?zqA}iqBXAc>AYiaVA9nF-&DZk|FJdoXixD-O0$WBz<!r)NYQC^quU#i~chLz#j3H>D$=4sV|KodVStZRi^Kb$G#o-FqJ~pphI1<Bzpg%hu-widAF-%zLVqrc@LcdtcNb{8y-KK6KAx;<L>PK;GhEOL0}pve#*i%jRblLy0>Spb>pMv*xsHg@Rb=@H`sT6wJY$ew+ejinImjk))xDff=+DHn>y2L?3sbgx6eucwVL?FcCK#%JhCnGgF+|$i<XZyab{aSMz!o5bZFkrn%#`URFKMhL4y;o-6ybmQ!Vb{0H5tl?0|3Qlc3q@zP9X;r%w<ES)NOsy{JewGXuw-T6SgUX9tpbB<CzHBf6j|EuPTJk;}L3QKQF;^8>lYp(s#H7UdGHSxjD#Tw~LhPH;89Zq0PTSgONqqcdTU?2&So8B0>=T8pgpkanhZyVd=DYrd}o*)R5F;wvWS7cy<%VNbwa(3EAB&)_<+V_D~U&Pe$t`g@Wp9<&tFb^e-UUqja<Bb-JE<R}P#Sw8hnC5mfOJg`^+WT_|>5~ZStR^;hYCQUsF4$HI{O7VB<qPivmg0O(xYD?7M*NN59Rl+<^pRlrQ%pM8nG$ul_=<Zc#05Do5?3ZM6ZhMOl)##b-pW$yKe~o{j_&xNTOH0-6AdSA?F@#~jOQJfAP^6X-iJ>e@V=*8(iyp*4*0N%oQoa*#GBi36z0JBX+?hm|^p{Z96+NVw)c^TfkV}-LEG&9^8`~>O%S<s*ijpatrb>Glv}YyCB%z;VRjA?3mH^60INt53vTjW~RIR?Za8mV@u!s}<xY=nzebx@}u#{;dU(csVzNy4KJ3lfjrQRsfjn~Pu=}jfeXCp5j+PPCJZ%2+Fc+S;4fSW@_M@U1^jXjxxHcqXvCBRT~9LKrLZCADHK)sU2SkOO+0w|Y|v8(cls;M58acBTF#GpqS5k+2II<miRKvr8|$SEQs1YUAh=B+O*!e~XZc&KGjlxnSWN)|);YC-KzhV9>Wx>_64cvyG(tGy(e+}-6y^7_1Ofk`V@;UgtsR?B=KncDdxgkVazz9S5T)`GLA19?u(Z9IGCxudz|Tsf1|`hc+Y6f6jdb@@_$0apTCZidW24-BdLeq}W1l`XYd+nqN15Zt)a8`Hm$<l24RUuluE+&Ipqvh+3BaG2twbILV$5OA2L&yZ>-6cRhjz{DZ@T7%aLxN!H1sH5l$l0^^ekgw{@R9@;Q<__*Wn9b!U5zn>)*ZTmKNHCwBnpdpD6~Rn({AAl-9EMccbB|@uUFLyC(@yIt&#Aj*_ie5PE+RQ9PBG+jpTHh5TdAb6vfe7KOzON+b!62S2!t-tMlx+Wlo0ukP^gXzgmvChG+oZ)6js~i+fp=g9H#)e>J^$bYEdQrpTz|#um', 'src/dataset.py': 'c-rL}U2o$y@?F1z&^}b^+S)iRio$m;knSegqKn-kF}4qep)wY2Gm#~gq~myl6#Wr>x~KbX{UtZU4~eqn>=rom_7Ee8EpcWzoNo>phG8JZ>_tWul}shoHW;-&!7DRD=A|HN$;joav;Gyl$;de?FBN0rEsR#YVmU8Z5(F2l%IS<r4K3NPYgWvdMfsWK`K5j32yeJr*0k))Ua(p6*^*qXd7eoUEg4lSTJ^}1OI3<(kH~D^BP%LzWK8&qE?_FUs&bfiF3J^Q6tFiiTM~?c4J7-fq!|;07mA5FFi8IV{SPuL^PDP%q;6Q68xS1elH9OL5uUL^@fppDDg#_mCD<)5*OJ^aA$eI4SrS$(aP8HW%$BrR025R&vf>4B1{{PS<bbcLQYa#~()}!DkpBGc;>RD}o~N&VxF91EzPO=_Mb2JqN^!%BMHuioX~7gFuv8$Gq@rp$3<!MOQ^E_$go=)OEu^@fyjZV*M=D8CxmOh}GI#-hRTgN`$r-P<iR6m{7VWsEw~I1Q=R61D`Cvt6gm2+8P!9#&rb^I4g2}8>mMJ&a?Dp9G1SQUvhLHZ5tQF5?f@bJwIaBbI*Nu?itb$*y6<C-<ZeCq#yAXk8Tg;MKS<LyufxbLD4}xdp&p&>D3V*QDmt}!kQc@fGf|V<##FmtE?XlL?j8i=!xdfNQ`;5p}$ME$G0>lbDUZSTD$pGTnVg&&K#xG$A+-(MSc+1G3KUXdS1+UZZPtV?6oTfirBBgJMri5`ufJ<2tl3X!x5JnIvx+%&{;i6j%6i&{@rx%(4P#ApEO@Yqg1u+zW%Z$$yMx+mA!H~xI-T35<TaVAwbJCjN89DwMBjI{g$Y7klK6!JtD+aZ#7=}|K$_>nWRl=Gfa=;q$a-;1AD1qlMO92#G6I+5`l4G*_c}BkZT2l;yA5YHD-<;Yw2riktIvJm&7w_JUgQw-4m?HZ*yJb1SbOc#I3mVj0iE@UPSS+5BTV`yo1(Blpr5Nuc{flQq0x=#cfhy-p|HcdoiH8?qO#0subO)0lF#+Q@W|19f67UsWD10Hw%ggsfCln2Q!PHt5VC<R@hR0dXXyKBm5ERm`hI~o$98ymSiH}s^kVx(y1fs?|holFU2<D|?x5i1*6bb~Drb&xOER<j&^YFfLiH8^WdX?~P*ZyFDmjGF1iC`d~nY@hv3-KS)6daL598fgZf=ggDj5Urrh~u2x0td!LU)Pk#`Gu*Cu!O2g3`b-^D_!Y9O|mLWwKQy{3y})CF#%vX3C}OWY2u+pkaCt~HVUz3q#}%yE2u5<C{E_~`<CWw2&FYOLmjBKN|7jVp~OwugL<;a%1xR>QKpuo!O@^cVA5?Vpnw=1YlJwV5*Hg4RWb+TQePd{%N@TTWATSU9|Jw;O`8<@A<72A_F6#ztOuD?X29Z~VVzvRg#Tm>3I(d5?Fvy%VQd0E79bF@?K0J^vOrpBH4ezmh@*>bM1?95p+jV7nb=4^^kjpSl~}Wy@JC6qT(X7&h^1cAkf4g%3ou1@&2SHmAM&9B7a?!yTaS*+ImWcqYRP6TP#TcbbWu2f0u&ZVSnm#++$Ik!s*xQmzuHE%Iw38MAaP1mR>tngX32!F4p24L)Q(BLsucGXfc6^*h0B5#kw@Q%r;^}=cQ(U;VH^2TRg}dCCd#Ox6HoesX^bjr-X*jE7PXAYQEhnE`;3!<IU{LhedlpyDP+$;e@wR;Ug*$9?rUL&HmrpDX-Mp2uL1C!Fa!Yau?6IG5)uvM9`|63UDq(iY8oITUc(IlnMZFN#S^-xp_W!_6ljLtC_I|shbLA{n7xlJC$L@xWoSQoY88ZTB&4Q5kLpTds8f3na{C+-@(>*vya=|>QnAWCULGhWzDJU}e0a1y(IbkA1yga8M+_U9h>n}ot4E+zK=yhj@U;qSBA0xQd)%un`NgpOizedsC@|!<28nU>3(6&v<UNLiQz3u`H@iGX3W(1s=BaY6B+symGH2wfTo;*a*Ck6Gkw6}`^d_yKFm0099#kC_re1Js18I+*deLlm7+rem$KKYs4H(nFLAV8)2yIcGTTR1e1U0N-SD+nUq;@m5vr*cIvs*3=*KDCIoJ+>u{QumU094?OK5vu#jzHBCp;Y50_X^aZr^2S@(FPQnC5`UQ_XDxrH((2Nie;<OCJA@&qK1#ree?UFhfIv_Yfc{G2D479#<$2zQC7O3b|zTwQU8}7ueESw)b1qRZ|lLp4B9YlZCHB-kFvQKr7_`0Lq3>7^(>bZQmfr$n}`u7E7%}J^p4BXAolQ0cGiCIR|vFa?dk&;4L;w9YhBP^bt7fx<FIY;Y9}bDBjfjdiwZTmSgd#v#YU3X5SFh|VH&k>+X_AN_}o6$9(a8Oo_B=a1~|s8RNJj8*#6`|Zbk2~VcNewid$o%83PjihC~CVJmY{h0G5WB7&IU_ME+~U-t{1!_5yPF6KSS;X8srwNSU!w$g%|scV{)it!G6UcTJ0e{4dPg-VONc)(gHq8eR{kog5n0EI`|!U)xUO>*zy|+{L)*jXprKC5s52{$6u+UC$AT-0CL79(fDteF_}8FfhRvJu;j=wu7|GI~N#tA?vRpk?R;t#g=?dKKcv{Tw!-&0$au@D$DEg+5OTiFD35fRH^F*8{WT=MDM&{w_%f`reiD;eR<y&_Qi=Wojl#P+_=0BfJeTJAx!K`__e9F?D@~<UWaNkvm9$V;YLwL(b#1Gn+m}HGF<j(cbA7JaG)sM<O@7yiu_8YPX=Cao82BK*jj1V3a&bBy^MflPyfPD>>ETJ??XbQ9rSh~@#P*gyf4LQkU;!YOFp|PpgI@<F0zHGY=k_%`QhzpdUA5GzPNmK_W!+4F)cmo{I0L?X}Vv7l)pWwdDCM56s@;r|66tc*EEm+*9D$CvMj|TJ&;uQd1A9ZOw6tI9d7UGQo7uv5Y{->$2MtGBT%b|xQ<fhU=^70&x8OTUqwEhejP+zEbExvOb?sF))7;r>3`W?^(SN&*bDHCtF75!lFvWtd4>-%l1<Uu9q6Eb<GOzHTbM{<GPLF5jPX3`1JDaoH^-R0j*_DqebBv^vSb?f@##g})p;5#u89vFbz8dbtY-Yt)m-naM%mRs(YeF;p=#A)jBCcDouahO<9>tB7C5@)X1f*Mph*mRa5qPUV70C-h+U0lz}h`x*`FPz>%r&rishjubQm6l?&Rq;^mGTs_7rXD@c3Ly*B~6$wcFzeszFengkVg`^3)Ekp=!pIH}=%Jk3Vh4cvZvQ<akOw$5~FL+&%G&?6>u6>`S^cz(}%u-myosjd%%`l!Iq>K$y0VB!?j|h*<~5jwZG=yP)uZ(Ym?PTRVG}+5vmZ#1(7{5caApaTh&?-GCRaa5agWchoe|>M_Ul$noDEHfLvudjHTKsVEm_WMcP|u0Yyweah-2Hf*u2#CEv1jty>DcF&f1W{;Fd`hUEK)i5{#U<|vPY!3(-f|Vgdc7emr4R|MCb>-G-T4%0LJ%TUOFB<9mbP&H0SVF6c6<K7K2l#O}7Q36_rMITpc$m2g3_R$-J=I@$bVns&`}mOLtod})t_0>}OgVr4>)(W2-telzrujXv{xBr>?7<yPKSr&gqHm66ME5frfy58(`-07|)+`L`Z#R{3Re-Uc>bGZ2s@);)qe1H8imjZJ6Q139_)a6(mL7R3n*dnTQbA8?nZu41`nI7j%^CA>ZW!;fd*70lUmcj+btsfn-*kxNq#mBe3Dl~;t{MBlqNAOn->)J%iP+)CiFnmU7ZDtOoPbwlbP+Ja(_JjD!sx=X!+U66nbEmqeW<ZF*W*a46Ui$#x@XeE9dkX3ruNaET9M*i4W84P?#}2tF7>@%eG?X(hPfWt?hRld(z*j&FqYx1g8DMB#^aIzH6*9lQ#*b2E>}nPrnxr%`nLC8=EMC#D)|SdYnFWgxm?fZIU}wF*@u9A{S9f#o_&B#DKMNKo&&rj#f#XWWp@^Qtj~hg8X|DlVKDWrNgv?u-$QO+E{lv*dNv!uk5X_jB${i^Vr`2s#_;qW>am%=KxEFa+UK(^JgjRVAz4}5?t_^DhW43iIx5+&N>B=dK`At&n1mXs^@4p->%t5lo54m?rW7=f&3GotD%BTs_Qwt5_E!GL6YeAYHm0NLHIE)kRuYaSEBc^A@w)5NJT@>;lcX!Q!f-)s{p$^4V1T?^sG81*L_XQ?h)Z0tzhQse8iM}-SX^xa', 'src/model_sdf.py': 'c-rL}+iu%N_FZ2wWgkMNG-JtjkP52;^(_fDH`vbZgJTdQawKt04Oz}mwrd#Z(|*8ifqm)M<xBRQxsXFjavW@1EMOssNX@y;xt|$(p4Ss){A`gYEQ>@k4U3!Jxbx|~WcjINtbE7hh<x0FFCyek#t7pvAeLJRQ{P0+sU$_ok~EfrP;#2iI7`T3DkSAG>m64-mTAt3e|mV_C(|-t0F*yxR4!-{kU0S6<xQWY3p!(@plKQQdarq$RS83af@RaKIOme4NO;N0behIg(VS#?d`Un6IcH>&O9{l4K=GW#m!bzh&DEOPETKFh36)eZ37aOgWaKI@FH=4vabA`zme~!F`HTSyNa2B|dg-FbOW6F4u#dUuMSp*P^4HhzUPgznPso^f&o1d~ma%7sn%7IGqy@usNm7nf!h*{As7K&q-@=p&R!V<6Xd(4$=Db><xPtJ){scOU=dG8Ja|Ob?zlKv|0-7Pfaj&P006q|cTBqrZ7=(l4m(7yVlU?(b#MUbg^<qG>oF>r~%d!aJ12U=7EQthN6d4N?%2{;w{_Mv$0eRIt>>nM!KRJp%oCf4H=)pdF-;&S2{_+_9!A|#4N9c3Pd&MK#doRwY5ab}|%YA;rvdRbVe48g##`-!W5(a@2r972U<YVLq5H?Xf4@h1~^?k`oj(IEnnT>Y>$QKFfay&2uR9saBEB$_06Y4ihKr+->V+^4Zp0b}R#$`$~zlOmNETNRc#J`8V^BlCWn~xRGd_#bwRE&>lCRjh{a@?mfp1;e>1+dlpR$)%qn~xw<k<plq&&tXr1A%09IL*r|S|&z?*N$>ZCaaP=itw*70JUXKm`Ia~X8UU4YmM_2CV<4V9DoTNS1_b3tyTD88{S%EEqqbt{~!SGKgf`SmR&ZO5>mP%LB1IqB7^}r8s&9@J|u7a{@0k`I4p4=&HkFl0T`uEzH9aZ@2G6Y?6u7f^DF*`+pC094C-wAN1Is30C}G25PM6nKfnuEU>K0$gA^n#(*lq!XY|mn`vOGJaPslJQOU>RPf^Gh^s#B9fs)?gbq$n`p$U+zEb@#><|7O`CIW_HP78KE99?WmGWfv;XY5VV44NcbdI6ysG-467HJp_zp6w<9RIqLppsDiDjd&LUNzw(R)yGzyoyPa^Rc2@NE5U=^Fn+)0MI{LtkqMS7f_*=^N}<nwVtV=~#NQ7F!Smf<xZ77f_Y>3Pw^kIf_mxCyBbN+}Bx^{(uTho@0ZSsL87=#@RGVUZ0zE23;s}CVLWu1Q9~R#efHkPScF+sS7VRPgDQ(iOC3{UzVv`+yz9mO`a2gR`Z|n0v|K;xtgTYWe4W0+j)x(Qmr;}<DNbp!In?w&RwV+Qip{@WA+rS7FDB#^Sz&k6zyBhdKOZt5nt}@1%9=Fy=IcWgy?X21<Q!%pqg0BZ`$#|$o(zaYxfjhJnLE%GTb;xy<QWe7v!mdFuiW#mT3_1uygMeypNeCk_fhbAH=U;w>nUGMJ%a&@C4M@c&dBqbLadQ<E9oGZ*SD0g`Hw8Z!bUK-S$7Qv9U$JyHm%?|7QhNbmo#=pyMHXscMH&m01^|21>o!CJMprfG`-`9vtOICTS7aDo(sTi_5pig9X%_j5P>T?l3O1!Yg8h5<)w&Mloa>?o4cOSjmdaag;Xu|!h&>*NtcLM-IcKA_JglTYvoaSbAfmM?vTWVno%a161pr^sY`fq4O7?cy>aXP-wJwEB#@pfcU=`1hZKG)x8~PQM=5>PZldT`rL%Vlzx(5Y(?wLy9T^MxuMOOgtqHZjDEgE%xRpW@;2!e;#04f!_?H~dSI{?Ejz)lBXr^TaQYBJdRb;u;X>5iP_=PVEH@^J0)z%I9f6l{Q}XSE89O<E#AK*0K~2aDdQM!D-b-mTU!Vzs$iZUyM)o~@tWg=M8){S^na<ZII*<u$#hHEgt?;?l9SG(WTuR$5Px$;m}Z676mcdwh~G>I6jd2#6-_1efhe@bjU}uOYX{n=Ks$pi_i~ueYa1VGVY`vJ6rH<O_3X2sxLQHy~Hc(j_Bvz-1-TG>FKX<3lGq#+88nKj&n|q{O2f1j5kTiXprN{!$`-xK>iVooI+g*U#spKxHKR|L8)M3)KV-f>(|9q=PCusOG^zJqw%+U=c~p72lpZBu$YrsIMFcEUa$pXlK`jzoj4P1Z-z))hEx$Ubx+GHF{KB8=f`2h6TC+4(tZuDe>#j*dh)2r&xbOQ7xQ?&7i8UXO@D7fkHfBTQ64J4`E@jLFY3^mcT(-*)J)-R4wDW1`-pXdo*r)P^fzX1cV+Io7F`Qj^u!W&Xk9kLuT;#Wg4^4`h<S1RjXf{z?^z<mC(^G3U=pVu$zPnzlN!Z=#r)xon#EQt4<`mxTrK`y>ET^J#KaPW@}^f-4A8%YsMOfW0n9779p<|pq0Mr2@1?Gid<OGk}epGtO8Fxw3C3Ee8@sA4Ut<u3hqb)PKaRyOZwriZz{Gij*hm+HFo~Zxz>10b0<ciy3)DP;TJ*tMCR_n*Y^lfnn2N$C0?{_C&~BUC-I)w8A^|M0@-MbLGY=opOL-iAuv;!Nfn=TDBu%j0G*z->LH)2n<MvrNQH89OQtT2nv0^V6q8ePUF6EMKBQG3KRANhQT2)6J{doGy0>@db;Hf|oo3_TY1vnx4?Drv0kB*?anBS0>x~3rx*&s*zQDknIp{#MnxDEkg8D=7I_URtU`H_T^z7(E%^VAC=Pru)1I0NY`E&~Jf}C#+gY95=(GSDWY%&o}8Z?N$Q>(HJoDE^lYu9r+A4?e5fi~F=$k169MV6XM=gE?$+zc^Y1Dq<-0#m8g9asZ8Pd^cTYYr;#N|2ae$q&fPvs;B3hAj%H#yCbQf48zLpEYuNuKEc1{M)~YB>*$imZGr9%EJ*kP;}9tCA}hAAKJ)K1TLRKuO3C@_qMQQj23qJs6$ZESklNd29us|>s}LD5(rGyO5ZkpmmNhw$3Bb_-Rnr_ETNuJ9ssJkZTx&06apKfP=ne#P<LI@61M4WT%Sj?(j)iaJL_j_@l`*54}P=W!PgT81bLL>sSe&d^z8A#CA$f<RnU>G?n5+J>!g6jvZC01$O;T*Qu;YyU+9BvXH=6^Rv3I9xT?yM7I$nI>D>7FxBqzV=wqP1F;MN<rxsqFfIAJXjmBHe_70vvsL@{r+jos8!D8uaF=6G!9hv2l+-ioMNY#uWNKcihixjdWRvl1|>&N<RrxTh#Lc-JzZp+K2xZD}RH-)$a%p6j%u59^~?EXnD!%D#JI4QX$&oZSd-@M6b0_Cxe+)TaKrLmG}Ce&XWgbg(0JTp(~V=a>g(c!A;G-K|4hGvQ9V(eP_pytxoELPf5lL`!mb}7ib+A;O?TAyKC?_906CQIB;#hiC<rUD71nW8sP!GrKF1><clTyxrDU@>6)T4ig~6ZzC6fRDWP_ca<?Q$_Zh)h{~i?uZ=g<0KdX1ol{EUd`sM>FAGi&j;{tT$3}oVN5zWei@z~9!L8>o*qR9uhxu9Wb?qpv7NkUT>4P2nQ6(YQb)k7sIa6#PrG#3>LeVHWqs&lb8m0$)pc`PxeQF`oiQ%*vVW(UsmmAhH$yqp6q>kcT<0wnWzw(ZGIF-v+JJU^TK~G<sayE!FsK$C@EuHj19YKw3ettU(^^-pz%VHK=<0ut3Ou!UH|4b$DvhOLipQ%h&gQtSP4*3L&YVGOlr;hN-qoN9u$HO}oF%3SZ3@9X$k0$Sni%+h8UDJlx+18LUYIg&{+_Sjiquf`*v8#Axo5NHnk1YUHh!%z-R-IgkYd|Sg;c$79k=#?5C<PCmcqpsp?eJI0h(!|2yw-|Z&1y7$6qv5KSb1i>;7eJ>cCJg!jk<|!SE?mlsYqpaY-u?HY53hxo0nPx>UW1I-&$7I0$G->P@Yi8?ZsC*H_%FJ34k<6|wanWGb^Vef?!;pL)t(H(rPLPN(Y6EF%t^gU$-V+l!!=O?eb~I&p10Sa8k}djACk)Ky6'}


def write_runtime() -> None:
    for rel, payload in RUNTIME_PAYLOADS.items():
        path = WORK / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(zlib.decompress(base64.b85decode(payload)).decode("utf-8"))


def main() -> None:
    write_runtime()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WORK) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    script = WORK / "inference.py"
    print(f"[cnn-exact] running {script}", flush=True)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(WORK), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"medali inference failed with rc={proc.returncode}")
    out = WORK / "submission.csv"
    if not out.exists():
        raise RuntimeError("medali inference did not create submission.csv")
    print(f"[cnn-exact] wrote {out}", flush=True)


if __name__ == "__main__":
    main()


# ---- guarded train-overlap override (pixiux/rogii-dual-pipeline-blend) ----
import shutil as _pre_shutil
from pathlib import Path as _PrePath

_pre_w = _PrePath("/kaggle/working") if _PrePath("/kaggle/working").exists() else _PrePath(".")
if (_pre_w / "submission.csv").exists():
    _pre_shutil.copyfile(_pre_w / "submission.csv", _pre_w / "submission_no_override.csv")

# Guarded train-overlap override, vendored verbatim from the public kernel
# pixiux/rogii-dual-pipeline-blend (121 votes, LB 7.519 vs 7.572 base).
# Reads submission.csv from the working dir, applies the guarded override,
# rewrites submission.csv. Appended to generated kernels by
# scripts/materialize_20260610_sp45heavy_candidates.py.

# Lesson learned: hidden rerun copies of "overlap" wells are NOT guaranteed to be
# same-version / row-aligned with their train copies - a blind 100% lookup can inject error.
# Guard: per well, validate the contacts reconstruction against the TEST copy's known
# prefix (TVT_input), interpolated BY MD (not row index); override only if rmse < 1 ft,
# and only rows whose MD lies inside the train copy's range. Otherwise keep the blend.
# By construction this is >= the plain blend: exact wells win, mismatched wells are skipped.
import os as _ov_os, glob as _ov_glob
import numpy as _ov_np, pandas as _ov_pd
from pathlib import Path as _OvPath

def _ov_tvt_from_contacts(hw_tr, tw_tr, ref_col="EGFDU"):
    tw_g = tw_tr.dropna(subset=["Geology"])
    ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    if _ov_np.isnan(ref_tvt):
        ref_col = tw_g["Geology"].iloc[0]; ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    offset = (hw_tr["TVT"] - (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]))).mean()
    return (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]) + offset).to_numpy(dtype=float)

try:
    _W = _OvPath("/kaggle/working") if _OvPath("/kaggle/working").exists() else _OvPath(".")
    _DATA = None
    for _c in [_OvPath("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
               _OvPath("/kaggle/input/rogii-wellbore-geology-prediction")]:
        if _c.exists() and (_c / "train").exists():
            _DATA = _c; break
    if _DATA is None:
        for _p in _ov_glob.glob("/kaggle/input/**/train/*__horizontal_well.csv", recursive=True):
            _DATA = _OvPath(_p).parent.parent; break
    _sub = _ov_pd.read_csv(_W / "submission.csv")
    _sub["well"] = _sub["id"].str[:8]; _sub["row_idx"] = _sub["id"].str[9:].astype(int)
    _pred = dict(zip(_sub["id"].astype(str), _sub["tvt"].astype(float)))
    _train_wells = set(_ov_os.path.basename(f).split("__")[0]
                       for f in _ov_glob.glob(str(_DATA / "train" / "*__horizontal_well.csv")))
    _n_ok = _n_skip = 0
    for _wid, _g in _sub.groupby("well"):
        if _wid not in _train_wells:
            continue
        try:
            _hw_te = _ov_pd.read_csv(_DATA / "test" / (_wid + "__horizontal_well.csv"))
            _hw_tr = _ov_pd.read_csv(_DATA / "train" / (_wid + "__horizontal_well.csv"))
            _tw_tr = _ov_pd.read_csv(_DATA / "train" / (_wid + "__typewell.csv"))
            _phys = _ov_tvt_from_contacts(_hw_tr, _tw_tr)
            _md_raw = _hw_tr["MD"].to_numpy(dtype=float)
            _m_fin = _ov_np.isfinite(_phys) & _ov_np.isfinite(_md_raw)
            if _m_fin.sum() < 100:
                print("override SKIP %s too few valid phys rows=%d" % (_wid, int(_m_fin.sum()))); _n_skip += 1; continue
            _o = _ov_np.argsort(_md_raw[_m_fin])
            _md_tr = _md_raw[_m_fin][_o]; _ph_tr = _phys[_m_fin][_o]
            # --- self-check: TEST copy known prefix (TVT_input) vs lookup, interpolated by MD ---
            _kn = _hw_te[_hw_te["TVT_input"].notna()]
            _kn = _kn[(_kn["MD"] >= _md_tr[0]) & (_kn["MD"] <= _md_tr[-1])]
            if len(_kn) < 50:
                print("override SKIP %s too few comparable known-prefix rows=%d" % (_wid, len(_kn))); _n_skip += 1; continue
            _rk = float(_ov_np.sqrt(_ov_np.mean(
                (_ov_np.interp(_kn["MD"].to_numpy(dtype=float), _md_tr, _ph_tr)
                 - _kn["TVT_input"].to_numpy(dtype=float)) ** 2)))
            if (not _ov_np.isfinite(_rk)) or _rk > 1.0:
                print("override SKIP %s known-prefix rmse=%.3f (train copy != test copy, keeping blend)" % (_wid, _rk))
                _n_skip += 1; continue
            # --- check passed -> override via MD interpolation (no row-index alignment), in-range rows only ---
            _md_te = _hw_te["MD"].to_numpy(dtype=float)
            _n_row = 0
            for _rid, _ri in zip(_g["id"].astype(str).values, _g["row_idx"].values):
                _ri = int(_ri)
                if 0 <= _ri < len(_md_te):
                    _m = float(_md_te[_ri])
                    if _md_tr[0] <= _m <= _md_tr[-1]:
                        _pred[_rid] = float(_ov_np.interp(_m, _md_tr, _ph_tr)); _n_row += 1
            print("override OK   %s known-prefix rmse=%.4f rows overridden=%d/%d" % (_wid, _rk, _n_row, len(_g)))
            _n_ok += 1
        except Exception as _e:
            print("override fallback %s: %s" % (_wid, _e)); _n_skip += 1
    _new = _sub["id"].astype(str).map(_pred).astype(float)
    assert _new.notna().all(), "override produced NaN, aborting"
    _sub["tvt"] = _new
    _sub[["id", "tvt"]].to_csv(_W / "submission.csv", index=False)
    print("GUARDED override done: overridden=%d skipped=%d (skipped = kept the blend)" % (_n_ok, _n_skip))
except Exception as _e:
    print("GUARDED override skipped entirely (kept the blend):", _e)
