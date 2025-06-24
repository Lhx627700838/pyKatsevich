clear
close all
% 参数
R = 610;               % source 轨道半径
P = -46;            % pitch
D = 1113;             % source-detector distance
lambda = linspace(0, 4*pi, 2016*2);  % lambda 取值？？？？怎么选出对应 lambda?
dlambda = lambda(2) - lambda(1);

% === 构造螺旋轨迹 ===
x_helix = R * cos(lambda);
y_helix = R * sin(lambda);
z_helix = P / (2*pi) * lambda;

x0 = 300; 
y0 = 100; 
z0 = -10;        % 重建点

% 定义 alpha*(lambda)
alpha_star = @(lambda) atan2(-x0*sin(lambda) + y0*cos(lambda), R - x0*cos(lambda) - y0*sin(lambda));

% 定义 w*(lambda)
w_star = @(lambda) D * cos(alpha_star(lambda)) ./ (R - x0*cos(lambda) - y0*sin(lambda)) .* (z0 - P/(2*pi)*lambda);
val_w_star = w_star(lambda);

% 定义 w_top and w_bottom
wtop = @(alpha)  (D*P)/(2*pi*R) * ((pi/2 - alpha) ./ cos(alpha));
wbot = @(alpha) -(D*P)/(2*pi*R) * ((pi/2 + alpha) ./ cos(alpha));
val_wtop = wtop(alpha_star(lambda));
val_wbot = wbot(alpha_star(lambda));

% 定义 rin(lambda) 和 rout(lambda)
rin = @(lambda) wtop(alpha_star(lambda)) - w_star(lambda);
rout = @(lambda) w_star(lambda) - wbot(alpha_star(lambda));

% 使用rin=0, rout=0估计 λ_i 和 λ_o
r_in = rin(lambda);
r_out = rout(lambda);

% === 找出r_in、r_out最接近0的位置，对应lambda_i、lambda_o ===
[~, idx_in] = min(abs(r_in));
[~, idx_out] = min(abs(r_out));
lambda_in = lambda(idx_in);
lambda_out = lambda(idx_out);
angle_lambda = abs(lambda_out-lambda_in)/pi*360;

% === 计算对应的source点位置 ===
a_in = [R*cos(lambda_in), R*sin(lambda_in), P/(2*pi)*lambda_in];
a_out = [R*cos(lambda_out), R*sin(lambda_out), P/(2*pi)*lambda_out];

fprintf('approximate between angle = %.4f grade \n', angle_lambda);
fprintf('approximate λ_i = %.4f rad, λ_o = %.4f rad\n', lambda_in, lambda_out);
fprintf('approximate a_i = [%.2f %.2f %.2f], a_o = [%.2f %.2f %.2f] \n', a_in, a_out);

% === 绘图 ===
figure(1);
plot3(x_helix, y_helix, z_helix, 'k--'); hold on;
plot3(x0, y0, z0, 'ro', 'MarkerSize', 8, 'DisplayName', 'Reconstruction Point');
plot3(a_in(1), a_in(2), a_in(3), 'bs', 'MarkerSize', 8, 'DisplayName', '\lambda_i (r_{in} \approx 0)');
plot3(a_out(1), a_out(2), a_out(3), 'gs', 'MarkerSize', 8, 'DisplayName', '\lambda_o (r_{out} \approx 0)');
plot3([a_in(1), a_out(1)], [a_in(2), a_out(2)], [a_in(3), a_out(3)], 'r-', ...
      'LineWidth', 2, 'DisplayName', '\pi-line');

xlabel('X'); ylabel('Y'); zlabel('Z');
legend;
axis equal; grid on;
title('Helical \pi-line through Reconstruction Point');

figure(2)
plot(lambda,r_in)
hold on
plot(lambda,r_out)
hold on
plot(lambda,val_w_star)
hold on
plot(lambda,val_wtop)
hold on
plot(lambda,val_wbot)
hold on
plot(lambda,zeros(length(lambda),1),'k--')
legend('r_in','r_out','val_w_star','val_wtop','val_wbot');

%% second point
x0 = 300; 
y0 = 300; 
z0 = -20;        % 重建点

% 定义 alpha*(lambda)
alpha_star = @(lambda) atan2(-x0*sin(lambda) + y0*cos(lambda), R - x0*cos(lambda) - y0*sin(lambda));

% 定义 w*(lambda)
w_star = @(lambda) D * cos(alpha_star(lambda)) ./ (R - x0*cos(lambda) - y0*sin(lambda)) .* (z0 - P/(2*pi)*lambda);
val_w_star = w_star(lambda);

% 定义 w_top and w_bottom
wtop = @(alpha)  (D*P)/(2*pi*R) * ((pi/2 - alpha) ./ cos(alpha));
wbot = @(alpha) -(D*P)/(2*pi*R) * ((pi/2 + alpha) ./ cos(alpha));
val_wtop = wtop(alpha_star(lambda));
val_wbot = wbot(alpha_star(lambda));

% 定义 rin(lambda) 和 rout(lambda)
rin = @(lambda) wtop(alpha_star(lambda)) - w_star(lambda);
rout = @(lambda) w_star(lambda) - wbot(alpha_star(lambda));

% 使用rin=0, rout=0估计 λ_i 和 λ_o
r_in = rin(lambda);
r_out = rout(lambda);

% === 找出r_in、r_out最接近0的位置，对应lambda_i、lambda_o ===
[~, idx_in] = min(abs(r_in));
[~, idx_out] = min(abs(r_out));
lambda_in = lambda(idx_in);
lambda_out = lambda(idx_out);
angle_lambda = abs(lambda_out-lambda_in)/pi*360;

% === 计算对应的source点位置 ===
a_in = [R*cos(lambda_in), R*sin(lambda_in), P/(2*pi)*lambda_in];
a_out = [R*cos(lambda_out), R*sin(lambda_out), P/(2*pi)*lambda_out];

fprintf('approximate between angle = %.4f grade \n', angle_lambda);
fprintf('approximate λ_i = %.4f rad, λ_o = %.4f rad\n', lambda_in, lambda_out);
fprintf('approximate a_i = [%.2f %.2f %.2f], a_o = [%.2f %.2f %.2f] \n', a_in, a_out);

% === 绘图 ===
figure(1);
plot3(x_helix, y_helix, z_helix, 'k--'); hold on;
plot3(x0, y0, z0, 'ro', 'MarkerSize', 8, 'DisplayName', 'Reconstruction Point');
plot3(a_in(1), a_in(2), a_in(3), 'bs', 'MarkerSize', 8, 'DisplayName', '\lambda_i (r_{in} \approx 0)');
plot3(a_out(1), a_out(2), a_out(3), 'gs', 'MarkerSize', 8, 'DisplayName', '\lambda_o (r_{out} \approx 0)');
plot3([a_in(1), a_out(1)], [a_in(2), a_out(2)], [a_in(3), a_out(3)], 'r-', ...
      'LineWidth', 2, 'DisplayName', '\pi-line');

xlabel('X'); ylabel('Y'); zlabel('Z');
legend;
axis equal; grid on;
title('Helical \pi-line through Reconstruction Point');