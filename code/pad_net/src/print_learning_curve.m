clc;
clear;

train = xlsread('LearningCurves.xlsx', 'train_msssimL2_10k');
val = xlsread('LearningCurves.xlsx', 'val_msssimL2_10k');

train_iter = train(:,1);
train_error = train(:,4);
val_iter = val(:,1);
val_error = val(:,4);

figure(1);
plot(train_iter, train_error);
hold on;
plot(val_iter, val_error,'LineWidth',2);
title('plot every data point');
xlabel('iteration');
ylabel('learning rate');
legend('train error', 'val error');

figure(2);
train_error_intv10=train_error(1:10:size(train_error));
train_iter_intv10=train_iter(1:10:size(train_iter));
plot(train_iter_intv10, train_error_intv10);
hold on;
plot(val_iter, val_error,'LineWidth',2);
title('plot every 100 data point');
xlabel('iteration');
ylabel('learning rate');
legend('train error', 'val error');