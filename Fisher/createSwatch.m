function swatch=createSwatch(xmin,xmax,ymin,ymax,num,varargin) 
xlen = abs(xmax - xmin);
ylen = abs(ymax - ymin);
if numel(varargin)>0 && isa(varargin{1},'function_handle') 
  f = varargin{1}; 
else
  f = @rand;
end 
swatch=[xlen*f(1,num)+min(xmax,xmin);ylen*f(1,num)+min(ymax,ymin)]; 
end
