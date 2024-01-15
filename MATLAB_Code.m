clc
clear all

%reading the csv file
data = readtable('winequality-white.csv');

%converting the tables into array
X = table2array(data(:, 1:end-1));
y = table2array(data(:, end));

%normalising the feature array to bring all the features on the same scale
X=normalize(X);

%merging the target values as 0 and 1. values from 3-6(bad) will be
%0,6-9(good) will be 1
y(y<=5) = 0;
y(y>5) = 1; 

%dividing the dataset into train and test sets
X_train = X(1:4000, :);
y_train = y(1:4000, :);
X_test = X(4001:end, :);
y_test = y(4001:end, :);

% defining the mlp
input_size = size(X_train, 2);
hidden_size = 7; 
output_size = 1;

%recording the time using tic-toc
tic
net = feedforwardnet(hidden_size);
%training the mlp with resilient back propagation
net.trainFcn = 'trainrp';
net.trainParam.showWindow = true;
net.trainParam.epochs = 250;
net.trainParam.lr = 0.4; 
net.divideFcn = 'dividerand';
[net,tr] = train(net, X_train', y_train');
train_time=toc;

% Predicting on train and test values
y_pred = round(net(X_test'));
x_pred=round(net(X_train'));

%finding train loss and accuracies for train and test for each epoch
train_acc = 1-perform(net,y_train', x_pred);
test_acc =1-perform(net,y_test', y_pred);
accuracy = sum(y_pred==y_test')/length(y_test);
train_loss = tr.perf;
fprintf('Accuracy: %f\n', accuracy);
fprintf('Training time: %.2f seconds\n', train_time);
test_acc_epochs = 1 - tr.tperf;

%plotting the accuracies and training loss
subplot(2,2,1);
plot(1:numel(test_acc_epochs), test_acc_epochs);
title('Test accuracy');
xlabel('Epoch');
ylabel('Test Accuracy');
subplot(2,2,2);
plot(train_loss);
title('Training Loss');
xlabel('Epochs');
ylabel('Loss');

