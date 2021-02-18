% Clean workspace
clear all; close all; clc
    
load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata
%%
L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

% averaging
Utnave = zeros(n,n,n);
for j=1:49
Un(:,:,:)=reshape(subdata(:,j),n,n,n);
% M = max(abs(Un),[],'all');
% close all, isosurface(X,Y,Z,abs(Un)/M,0.7)
% axis([-20 20 -20 20 -20 20]), grid on, drawnow
% pause(1)
Utn = fftn(Un);
Utnave = Utnave + Utn;
end

Utnave = fftshift(abs(Utnave) / 49);
Mk = max(abs(Utnave),[],'all');
close all, isosurface(Kx,Ky,Kz,Utnave/Mk,0.7);
grid on, drawnow
title('Center Frequency');
xlabel('x-frequency (Kx)');
ylabel('y-frequency (Ky)');
zlabel('z-frequency(Kz)');

%% Filtering
% The range of the center frequency is about from x[4.4,
% 5.5],y[-7.5,-6.5],z[1.4,2.6]
% Select the filter function:
tau = 0.5;
kx0 = 5;
ky0 = -7;
kz0 = 2;
filter = exp(-tau*(Kx - kx0).^2) .* exp(-tau*(Ky - ky0).^2) .*  exp(-tau*(Kz - kz0).^2);
location = zeros(49,3);

for j=1:49
Un(:,:,:)=reshape(subdata(:,j),n,n,n);
Utn = fftshift(fftn(Un));
Utnf = filter.*Utn; % Apply the filter to the signal in frequency space
Unf = ifftn(Utnf); % inverse fourier transform to time domian. 
maxCor = max(abs(Unf),[],'all');
[xt,yt,zt] = ind2sub(size(Unf),find(abs(Unf)== maxCor));
location(j,:) = [X(xt, yt, zt), Y(xt, yt, zt), Z(xt, yt, zt)];
end
plot3(location(:, 1), location(:, 2), location(:, 3));
title('Submarine Track');
xlabel('x');
ylabel('y');
zlabel('z');
set(gca,'FontSize',14)
grid on;


%% table of x and y coord
xy = table(location(:,1), location(:,2));











