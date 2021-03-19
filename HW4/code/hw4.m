clear all; close all; clc;
%% loading
[images, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');

%% SVD Analysis
% for i = 1:20 % show images
%     imshow(images(:, :, i));
% end
[x, y, z] = size(images);
I = im2double(reshape(images, [x * y, z]));
[U,S,V] = svd(I, 'econ');

%% Plot singular values
figure(1)
plot(diag(S),'o','Linewidth',2)
title('Singular Value Spectrum')
xlabel('Principal Components')
ylabel('Singular values')
set(gca,'Fontsize', 14)
% set(gca,'Fontsize',16,'Xlim',[0 80]) % plot first 80 singular values
%% Projection onto V
figure(2)
projection = S * V';
for i = 0:9
    index = find(labels == i); % find which index of data contains label i;
    plot3(projection(1, index), projection(3, index), projection(5, index), "o");
    hold on;
end
xlabel('1st V-mode'), ylabel('3rd V-Mode'), zlabel('5th V-Mode')
title('Projection on 1,3,5 V-modes')
legend('label1', 'label2', 'label3', 'label4','label5', 'label6',...
    'labe7', 'labe8', 'label9')
set(gca,'Fontsize', 14)


%% load test data
[test_images, test_labels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
[x, y, z] = size(test_images);
test_I = im2double(reshape(test_images, [x * y, z]));

%% 2 digits LDA
feature = 20;
label1 = 0; label2 = 9;
index1 = find(labels == label1); index2 = find(labels == label2);
data1 = I(:, index1); data2 = I(:, index2);
[U2,S2,V2,threshold,w,sort1,sort2] = dc_trainer(data1, data2,feature);
test_index1 = find(test_labels == 0); test_data1 = test_I(:, test_index1);
test_index2 = find(test_labels == 9); test_data2 = test_I(:, test_index2);

TestSet = [test_data1, test_data2];
TestNum = size(TestSet,2);
TestLabel = zeros(1,TestNum);
TestLabel(1, size(test_data1, 2) + 1:end) = 1;
TestMat = U2'*TestSet; % PCA projection
pval = w'*TestMat;

ResVec = (pval > threshold); err = abs(ResVec - TestLabel);
errNum = sum(err);
success_rate = 1 - errNum / TestNum; 


%% random 2 digit LDA
success_rate_lowest = 1;
group1 = 0; 
group2 = 1; 
success_rate_highest = 0;
group1_high = 0; 
group2_high = 1;
for i = 0:9
    for j = i + 1 : 9
        label1 = i; label2 = j;
        index1 = find(labels == label1); index2 = find(labels == label2);
        data1 = I(:, index1); data2 = I(:, index2);
        [U2,S2,V2,threshod,w,sortdog,sortcat] = dc_trainer(data1, data2,feature);
        test_index1 = find(test_labels == 0); test_data1 = test_I(:, test_index1);
        test_index2 = find(test_labels == 9); test_data2 = test_I(:, test_index2);

        TestSet = [test_data1, test_data2];
        TestNum = size(TestSet,2);
        TestLabel = zeros(1,TestNum);
        TestLabel(1, size(test_data1, 2) + 1:end) = 1;
        TestMat = U2'*TestSet; % PCA projection
        pval = w'*TestMat;

        ResVec = (pval > threshold); err = abs(ResVec - TestLabel);
        errNum = sum(err);
        success_rate = 1 - errNum / TestNum; 
            if success_rate < success_rate_lowest
        group1 = i; 
        group2 = j; 
        success_rate_lowest = success_rate;
    end
    if success_rate > success_rate_highest
        group1_high = i; 
        group2_high = j; 
        success_rate_highest = success_rate;
    end
    end
end

%% Three digits LDA
label1 = 0; label2 = 1; label3 = 2; 
index1 = find(labels == label1); index2 = find(labels == label2); index3 = find(labels == label3);
data1 = I(:, index1); data2 = I(:, index2); data3 = I(:, index3);
[U5,S5,V5,threshold1,threshold2,w,max_ind,min_ind] = dg3_trainer(data1,data2,data3,feature);

test_index1 = find(test_labels == 0); test_data1 = test_I(:, test_index1);
test_index2 = find(test_labels == 1); test_data2 = test_I(:, test_index2);
test_index3 = find(test_labels == 2); test_data3 = test_I(:, test_index3);

TestSet = [test_data1, test_data2, test_data3];
TestNum = size(TestSet,2);
TestLabel = zeros(1,TestNum);
TestLabel(1, size(test_data1, 2) + 1:end) = 1;
TestMat = U2'*TestSet; % PCA projection
pval = w'*TestMat;

ResVec = (pval > threshold); err = abs(ResVec - TestLabel);
errNum = sum(err);
success_rate = 1 - errNum / TestNum; 


%% SVM and Decision tree
% classification tree on fisheriris data
load fisheriris;
tree = fitctree(meas,species,'MaxNumSplits',3,'CrossVal','on');
view(tree.Trained{1},'Mode','graph');
classError = kfoldLoss(tree)

% SVM classifier with training data, labels and test set
Mdl = fitcsvm(xtrain,label);
test_labels = predict(Mdl,test);

%%
tree = fitctree(projection', labels, 'CrossVal', 'on');
classError = kfoldLoss(tree); % tree error  0.1755
tree_sucrate = 1 - classError; % 0.8245
label1 = 0; label2 = 9; % easy
index1 = find(labels == label1); index2 = find(labels == label2);
data1 = I(:, index1); data2 = I(:, index2);
label1 = 1; label2 = 9; % hard
index1 = find(labels == label1); index2 = find(labels == label2);
data1 = I(:, index1); data2 = I(:, index2);
% SVM
Mdl =  fitcecoc(projection(1:50, :)', labels);
predict_labels_svm = predict(Mdl, TestMat');
test_labels = predict(Mdl,test1);














