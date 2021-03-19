clear all; close all; clc

% HW3 Leqi Wang
%% load and play vedio
load('cam1_1.mat') %  case1
load('cam2_1.mat') 
load('cam3_1.mat')
% load('cam1_2.mat') % Case1
% load('cam2_2.mat')
% load('cam3_2.mat')
% load('cam1_3.mat') % Case1
% load('cam2_3.mat')
% load('cam3_3.mat')
% load('cam1_4.mat') % Case1
% load('cam2_4.mat')
% load('cam3_4.mat')
vids = {vidFrames1_1,vidFrames2_1(:,:,:,10:end),vidFrames3_1}; % case 1: shift graph to make peak aligned
% vids = {vidFrames1_2(:,:,:,10:end),vidFrames2_2,vidFrames3_2}; % Case1
% vids = {vidFrames1_3,vidFrames2_3,vidFrames3_3}; % Case1
% vids = {vidFrames1_4,vidFrames2_4,vidFrames3_4}; % case 4
% implay(vids{1})
% implay(vids{2}
% implay(vids{3})

%% Crop
prediction1 = [308.5 218.5 28 205]; % for cam1_1
prediction2 = [229.5 66.5 59 295]; % for cam2_1
prediction3 = [139.5 246.5 258 302]; % for cam3_1
% prediction1 = [305.5 198.5 139 212]; % for cam1_2
% prediction2 = [200.5 62.5 283 362]; % for cam2_2
% prediction3 = [130.5 307.5 228 267]; % for cam3_2
% prediction1 = [305.5 198.5 139 212]; % for cam1_3
% prediction2 = [200.5 62.5 283 362]; % for cam2_3
% prediction3 = [134.5 161.5 240 418]; % for cam3_3
% prediction1 = [305.5 198.5 139 212]; % for cam1_4
% prediction2 = [200.5 62.5 283 362]; % for cam2_4
% prediction3 = [134.5 161.5 240 418]; % for cam3_4

predictions = {prediction1, prediction2, prediction3};
numFrames = min(min(size(vids{1},4), size(vids{2},4)), size(vids{3},4)); 
data1 = zeros(2, numFrames);
data2 = zeros(2, numFrames);
data3 = zeros(2, numFrames);
data = {data1, data2, data3};
% [J,rect] = imcrop(I');
%% extract x_1, y_1 ~ x_3, y_3 to construct covariance matrix;
for i = 1:3
    vid = vids{i};
    prediction = predictions{i};
    for t = 1:numFrames
        I = rgb2gray(vid(:,:,:,t));
        if(i == 3)
            I = I';
        end
%         I = double(imcrop(I, prediction)); % crop image
        I = imcrop(I, prediction);
        target = max(max(I)); % grayscale of the white spot, as target
        [vert, hori] = ind2sub(size(I), find(I == target));
        data{i}(1,t) = mean(vert); % center of mass
%         if data{i}(1,t) < 90 % the light dot disappears
%             data{i}(1,t) = data{i}(1,t - 1);
%         end
        data{i}(2,t) = mean(hori);
        % imshow(I); drawnow
    end
    data{i}(1,:) = data{i}(1,:) - mean(data{i}(1,:));
    data{i}(2,:) = data{i}(2,:) - mean(data{i}(2,:));
    data{i}(1,:) = data{i}(1,:) / sqrt(numFrames - 1);
    data{i}(2,:) = data{i}(2,:) / sqrt(numFrames - 1);
    t = linspace(0, numFrames, numFrames);
    figure(1)
%     subplot(2, 2, 1)
    subplot(2, 2, 1)
    title('Case1: 3 camara z-direction without PCA', 'fontsize', 14)
    plot(t, data{i}(1, :))
    xlabel('Frames')
    ylabel('relative position in pixel(scaled)')
    hold on
    legend('cam1', 'cam2', 'cam3')
    
    subplot(2,2,3) 
    title('Case1: 3 camara x-y plane direction without PCA', 'fontsize', 14)
    plot(t, data{i}(2, :))
    xlabel('Frames')
    ylabel('relative position in pixel(scaled)')
    legend('cam1', 'cam2', 'cam3')
    axis([0 numFrames -10 10])
    hold on
end

%% Case PCA analysis
X = [data{1}(1, :); data{1}(2, :); data{2}(1, :);data{2}(2, :); data{3}(1, :); data{3}(2, :)];
[u,s,v] = svd(X, 'econ');
lambda = diag(s).^2;
subplot(2, 2, 2) % PCA
plot(t,v(:,1:2)')
legend('PC1', 'PC2', 'PC3')
title('Case1 Principal Component Oscillation', 'fontsize', 14)
xlabel('Frames')
ylabel('relative position in pixel(scaled)')

%%
subplot(2, 2, 4)
plot(1:6, lambda/sum(lambda), 'o')
title('Case1: Energy of Principal Component','fontsize', 14)
xlabel('#principal component')
xticks([1 2 3 4 5 6])
xticklabels({'Sigma1','Sigma2','Sigma3','Sigma4','Sigma5','Sigma6'})
ylabel('energy percentage')

hold off;

