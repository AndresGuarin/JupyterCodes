clc
clear all 
%
phc=fundamentalPhysicalConstantsFromNIST();
me=phc.electron_mass.value;
q=abs(phc.electron_charge_to_mass_quotient.value*me);
eps_0=phc.electric_constant.value;
ct_k=q/(4*pi*eps_0);
x1=-15:1:15;
a=10;
[x,y]=meshgrid(x1);
f_1=@(x,y)(1./sqrt(x.^2+(y-a).^2))*0;
f_2=@(x,y)(1./sqrt(x.^2+(y+a).^2))*0;
f_3=@(x,y)(1./sqrt((x-a).^2+(y).^2));
f_4=@(x,y)(1./sqrt((x+a).^2+(y).^2));
f_5=@(x,y)(1./sqrt((x-a*cosd(30)).^2+(y-a*sind(30)).^2))*0;
f_6=@(x,y)(1./sqrt((x-a*cosd(60)).^2+(y-a*sind(60)).^2))*0;
f_7=@(x,y)(1./sqrt((x-a*cosd(120)).^2+(y-a*sind(60)).^2))*0;
f_8=@(x,y)(1./sqrt((x-a*cosd(150)).^2+(y-a*sind(150)).^2))*0;
f_9=@(x,y)(1./sqrt((x-a*cosd(210)).^2+(y-a*sind(210)).^2))*0;
f_10=@(x,y)(1./sqrt((x-a*cosd(240)).^2+(y-a*sind(240)).^2))*0;
f_11=@(x,y)(1./sqrt((x-a*cosd(300)).^2+(y-a*sind(300)).^2))*0;
f_12=@(x,y)(1./sqrt((x-a*cosd(330)).^2+(y-a*sind(330)).^2))*0;

v=@(x,y)(f_1(x,y)+f_2(x,y)+f_3(x,y)-f_4(x,y)+f_5(x,y)+f_6(x,y)+f_7(x,y)+f_8(x,y)+f_9(x,y)+f_10(x,y)+f_11(x,y)+f_12(x,y));
hold on
axis([-15 15 -15 15])
contour(x,y,v(x,y),20)
Ex=-(x.*(-(f_1(x,y)).^3 -(f_2(x,y)).^3)+((x-a).*(-f_3(x,y)).^3 +(x+a).*(f_4(x,y)).^3)+((x-a*cosd(30)).*(f_5(x,y)).^3 )+(x-a*cosd(60)).*(f_6(x,y)).^3)-((x-a*cosd(120)).*(f_7(x,y)).^3 +(x-a*cosd(150)).*(f_8(x,y)).^3);
Ey=-(-(y-a).*(f_1(x,y)).^3 -(y+a).*(f_2(x,y)).^3)-y.*((-f_3(x,y)).^3 +(f_4(x,y)).^3)-((y-(a*sind(30))).*(f_5(x,y)).^3)-((y-(a*sind(60))).*(f_6(x,y)).^3)-((y-(a*sind(120))).*(f_7(x,y)).^3)-((y-(a*sind(150))).*(f_8(x,y)).^3);
magni_E=sqrt(Ex.^2+Ey.^2);
quiver(x,y,(-1.*Ex)./magni_E,(-1.*Ey)./magni_E,'color','r')
hold off
print -dmeta