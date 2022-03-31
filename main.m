function main()
load("CSE847/data/alzheimers/ad_data.mat");

pars = [1e-8,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0];
aucs = zeros(length(pars),1);

for ix=1:length(pars)
    [w,c] = logistic_l1_train(X_train, y_train, pars(ix));
    predictions = ((X_test * w + c) > 0) * 2 - 1;
    [X,Y,T,AUC] = perfcurve(y_test, predictions, 1);
    aucs(ix) = AUC;
end
plot(pars,aucs);

end