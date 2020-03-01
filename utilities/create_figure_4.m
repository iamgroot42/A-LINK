data = [60.43 83.25 88.08; 49.91 65.73 70.65; 78.39 88.68 92.75; 67.15 72.88 84.46];
bar(data);

a = (1:size(data,1)).';
x = [a-0.23 a a+0.23];
for k=1:size(data,1)
    for m = 1:size(data,2)
        text(x(k,m),data(k,m),num2str(data(k,m),'%0.2f'), 'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize', 16);
    end
end


xtl = {{'L-CSSE';'  GAR_{0.1%}'} {'L-CSSE'; '  GAR_{0.01%}'} {'DenseNet'; '  GAR_{0.1%}'} {'DenseNet'; '  GAR_{0.01%}'}};
h = my_xticklabels(gca,[1 2 3 4],xtl)

% names = {'L-CSSE GAR_{0.1%}', 'L-CSSE GAR_{0.01%}', 'DenseNet GAR_{0.1%}', 'DenseNet GAR_{0.01%}'};
% set(gca,'xticklabel', names);
ylim([40 100]);
ylabel('Genuine Accept Rate');
legend({'M_{1}', 'A-LINK', 'A2-LINK'});
set(gca,'FontSize',22);
