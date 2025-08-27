function [xx,density,G,Z] = findPointDensity(points,sigma,numPoints,rangeVals)

    if nargin < 3 || isempty(numPoints)
        numPoints = [1001 1001];
    end

    % Allow scalar or two-element vector for grid size
    if numel(numPoints) == 1
        numPoints = [numPoints numPoints];
    end
    
    % Ensure odd grid sizes for both dimensions
    for k = 1:2
        if mod(numPoints(k),2) == 0
            numPoints(k) = numPoints(k) + 1;
        end
    end
    
    if nargin < 4 || isempty(rangeVals)
        rangeVals = [-110 110];
    end

    % Create grids with potentially different resolutions per axis
    xx = linspace(rangeVals(1),rangeVals(2),numPoints(1));
    yy = linspace(rangeVals(1),rangeVals(2),numPoints(2));
    [XX,YY] = meshgrid(xx,yy);
    dx = xx(2) - xx(1);
    
    G = exp(-.5.*(XX.^2 + YY.^2)./sigma^2) ./ (2*pi*sigma^2);
    
    Z = hist3(points,{xx,yy});
    Z = Z ./ (sum(Z(:)));
    
    density = fftshift(real(ifft2(fft2(G).*fft2(Z))))';
    density(density<0) = 0;
    
    %imagesc(xx,yy,density)
    %axis equal tight
    %set(gca,'ydir','normal');