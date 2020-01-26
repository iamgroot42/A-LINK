data = [56.01 56.02 56.88; 75.62 80.96 81.57; 88.21 85.98 87.05; 89.92 86.89 87.60; 93.77 89.14 90.01; 90.66 88.00 88.72];
bar(data);
xtl = {{'M_{1}';''} {'M_{2} before'; 'A2-LINK'} {'M_{2} without'; 'A2-LINK'} {'M_{2} with A2-LINK:'; 'no noise'} {'M_{2} with A2-LINK:'; 'mixture'} {'A-LINK'}};
h = my_xticklabels(gca,[1 2 3 4 5 6],xtl)
ylim([50 100]);
ylabel('Accuracy (%)');
legend({'Impersonation', 'Obfuscation', 'Overall'});
set(gca,'FontSize',20);
grid on;
grid minor;
