clear; clc;
%定义两类样本的空间范围 
x1min=2;x1max=6; 
y1min=-4,y1max=0; 
x2min=6,x2max=10; 
y2min=2,y2max=6; 
%产生两类2D空间的样本 
c1=createSwatch(x1min,x1max,y1min,y1max,100); 
c2=createSwatch(x2min,x2max,y2min,y2max,80); 
%获取最佳投影方向 
w=fisher_w(c1,c2); 
%计算将样本投影到最佳方向上以后的新坐标 
cm1=c1(1,:)*w(1)+c1(2,:)*w(2); 
cm2=c2(1,:)*w(1)+c2(2,:)*w(2); 
cc1=[w(1)*cm1;w(2)*cm1]; 
cc2=[w(1)*cm2;w(2)*cm2]; 
%打开图形窗口 
figure; 
%绘制多图 
hold on; 
%绘制第一类的样本 
plot(c1(1,:),c1(2,:),'rp'); 
%绘制第二类的样本 
plot(c2(1,:),c2(2,:),'bp'); 
%绘制第一类样本投影到最佳方向上的点 
plot(cc1(1,:),cc1(2,:),'r+'); 
%绘制第二类样本投影到最佳方向上的点 
plot(cc2(1,:),cc2(2,:),'b+'); 
w=15*w; 
%画出最佳方向 
line([-w(1),w(1)],[-w(2),w(2)],'color','k'); 
axis([-10,15,-10,15]); 
grid on; 
hold off; 