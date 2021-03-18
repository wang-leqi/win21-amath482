clear all; close all; clc;

%%
filename = 'ski_drop_low.mp4'; 
% filename = 'monte_carlo_low.mp4';
v = VideoReader(filename);
numFrames = v.NumFrames; % time(s)
m = 1; 
dt = 10; 
prediction = [230 140 484 395]; % crop ski video;
for i = 1:dt:numFrames % construct X
    frames = im2double(rgb2gray(read(v,i))); % read(v,i) to read every frames
    frames = imcrop(frames, prediction); 
    [x, y] = size(frames);
    frames = frames(1: x, 1: y); 
    frames = reshape(frames, x * y,1);
    X(:, m) = frames;
    m = m + 1; 
end
m = m - 1; 
n = x * y;
X1 = X(:, 1:end - 1); % data matrix 
X2 = X(:, 2:end);
[U, S, V] = svd(X1, 'econ');

figure(1);
plot(diag(S), 'o', 'Linewidth', 2);
title("Singular Values, 2nd Video", 'FontSize', 14);
xlabel('singular values');
ylabel('energy');

r = 20; % keep ranks
U_trunc = U(:, 1:r); S_trunc = S(1:r, 1:r); V_trunc = V(:, 1:r);
A_tilde = U_trunc' * X2 * V_trunc / S_trunc;
[W, D] = eig(A_tilde); % W = basis;
Phi = X2 * V_trunc / S_trunc * W; % DMD eigen vectors

lambda = diag(D);
omega = log(lambda) / dt;

%%
bg = find(abs(omega) < 1e-2); % calculate p for background
fg = setdiff(1:r, bg); % foreground(j != p)

omega_bg = omega(bg); % background 
Phi_bg = Phi(:,bg); 
omega_fg = omega(fg); % foreground
Phi_fg = Phi(:,fg);

%% background
x1 = X(:, 1); % use first frame as initial condition
bp = transpose(Phi_bg) * x1; 
% reconstruct background
X_bg = zeros(x * y, m); 
for i = 1:m
    t = i * dt; % which frame
    X_bg(:, i) = bp * Phi_bg * exp(omega_bg * t);
end
% X_bg = Phi_bg * diag(X_bg); % modes projection
X_bg_reshape = reshape(X_bg, x, y, m); 

%% foreground
X_fg = X - abs(X_bg); 

% isolation of foreground by R
R = X_fg .* (X_fg < 0); 
X_fg = X_fg - R - R; 
X_fg_reshape = reshape(X_fg, x, y, m);

%% gen pics for monte_carlo
imwrite(X_bg_reshape(:, :, 9), 'car_bg1.jpg'); 
imwrite(X_bg_reshape(:, :, 20), 'car_bg2.jpg'); 
imwrite(X_bg_reshape(:, :, 38), 'car_bg3.jpg'); 
imwrite(X_fg_reshape(:, :, 9), 'car_fg1.jpg'); 
imwrite(X_fg_reshape(:, :, 20), 'car_fg2.jpg'); 
imwrite(X_fg_reshape(:, :, 38), 'car_fg3.jpg'); 

%% gen pics for ski
imwrite(X_bg_reshape(:, :, 10), 'ski_bg1.jpg'); 
imwrite(X_bg_reshape(:, :, 20), 'ski_bg2.jpg'); 
imwrite(X_bg_reshape(:, :, 42), 'ski_bg3.jpg'); 
imwrite(X_fg_reshape(:, :, 10), 'ski_fg1.jpg'); 
imwrite(X_fg_reshape(:, :, 20), 'ski_fg2.jpg'); 
imwrite(X_fg_reshape(:, :, 42), 'ski_fg3.jpg'); 
%%
for i = 1:m
    fg_t = X_fg_reshape(:, :, i); % test frame background @ t
    imshow(fg_t)
end






