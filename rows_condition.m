%------------------------------
% Input parameters
%------------------------------
P = 46.0;                         % Pitch (mm)
R = 610;                          % Distance from source to iso-center (mm)
R0 = 1113;                        % Distance from source to detector (mm)
D = R;                            % D = R (if not otherwise specified)
alpha_m = 0.426235;               % Half fan angle (radians)
dw = 0.4008525174825175;          % Detector row thickness (mm)
du = 0.689741074909091;           % Detector element spacing (mm)
n_u = 1376;                       % Number of detector columns

%------------------------------
% Compute u range
%------------------------------
u_m = D * tan(alpha_m);                          % Max u (symmetric about 0)
u = linspace(-u_m, u_m, n_u);                    % 1376 detector columns

%------------------------------
% Compute w_top(u) and w_bottom(u)
%------------------------------
w_top = (P ./ (2 * pi * R0 * D)) .* ((u.^2 + D^2) .* (pi/2 - atan(u ./ D)));
w_bottom = -(P ./ (2 * pi * R0 * D)) .* ((u.^2 + D^2) .* (pi/2 + atan(u ./ D)));

%------------------------------
% Compute N_rows and M
%------------------------------
term = (u_m^2 + D^2) * (pi/2 + atan(u_m / D));
N_rows = 1 + (P / (pi * R0 * D * dw)) * term;

factor = P * D / (N_rows * R0 * dw);  % Pitch factor
M = 0.5 * N_rows * (1 + (pi/2 + alpha_m) * tan(alpha_m));

%------------------------------
% Output
%------------------------------
fprintf('Minimum required detector rows: %.2f\n', N_rows);
fprintf('Maximum pitch factor: %.2f\n', factor);
fprintf('Recommended number of kappa-curves M: %.2f (rounded: %d)\n', M, round(M));

%------------------------------
% Optional plot
%------------------------------
figure;
plot(u, w_top, 'r', u, w_bottom, 'b');
xlabel('u (mm)');
ylabel('w (mm)');
title('Tam–Danielsson window boundaries');
legend('w_{top}(u)', 'w_{bottom}(u)');
grid on;
