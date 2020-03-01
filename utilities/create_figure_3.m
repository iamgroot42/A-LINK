data = [56.01 56.02 56.88; 75.62 80.96 81.57; 88.21 85.98 87.05; 89.92 86.89 87.60; 93.77 89.14 90.01; 90.66 88.00 88.72];
bar(data);

a = (1:size(data,1)).';
x = [a-0.23 a a+0.23];
for k=1:size(data,1)
    for m = 1:size(data,2)
        text(x(k,m),data(k,m),num2str(data(k,m),'%0.2f'), 'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize', 16);
    end
end

xtl = {{'M_{1}';''} {'M_{2} before'; 'A2-LINK'} {'M_{2} without'; 'A2-LINK'} {'M_{2} with A2-LINK:'; 'no noise'} {'M_{2} with A2-LINK:'; 'mixture'} {'A-LINK'}};
set(gca,'FontSize',22);
h = my_xticklabels(gca,[1 2 3 4 5 6],xtl)
ylim([50 100]);
ylabel('Genuine Accept Rate');
legend({'Impersonation', 'Obfuscation', 'Overall'});
set(gca,'FontSize',22);
