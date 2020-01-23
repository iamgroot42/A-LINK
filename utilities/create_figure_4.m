data = [60.43 83.25 88.08; 49.91 65.73 70.65; 78.39 88.68 92.75; 67.15 72.88 84.46];
bar(data);
names = {'L-CSSE GAR_{0.1}%', 'L-CSSE GAR_{0.01}%', 'DenseNet GAR_{0.1}%', 'DenseNet GAR_{0.01}%'};
set(gca,'xticklabel', names);
ylim([40 100]);
ylabel('Accuracy (%)');
legend({'M_{1}', 'A-LINK', 'A2-LINK'});
