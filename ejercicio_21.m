% ============================
% DATOS
% ============================
X = [0 0;
     2 1;
     3 3;
     4 4;
     1 3];      % puntos (x1,x2)

y = [0; 0; 1; 1; 1];   % clases
N = length(y);

% ============================
% PARAMETROS
% ============================
eta = 0.1;
epocas = 200;

w = zeros(2,1);   % pesos iniciales
b = 0;

sigmoid = @(z) 1./(1+exp(-z));

% ============================
% ENTRENAMIENTO
% ============================
for k = 1:epocas
    
    z = X*w + b;
    y_hat = sigmoid(z);
    
    % gradientes
    error = y_hat - y;
    dw = (1/N) * X' * (error .* y_hat .* (1 - y_hat));
    db = (1/N) * sum(error .* y_hat .* (1 - y_hat));
    
    % actualizacion
    w = w - eta*dw;
    b = b - eta*db;
end

% ============================
% MALLA DEL PLANO
% ============================
paso = 0.5;
[x1g, x2g] = meshgrid(-20:paso:20, -20:paso:20);

Zg = w(1)*x1g + w(2)*x2g + b;
Yg = sigmoid(Zg);

figure
hold on

% malla clasificada
scatter(x1g(Yg<0.5), x2g(Yg<0.5), 10, 'b', 'filled')
scatter(x1g(Yg>=0.5), x2g(Yg>=0.5), 10, 'g', 'filled')

% puntos reales
scatter(X(y==0,1), X(y==0,2), 80, 'b', 'filled')
scatter(X(y==1,1), X(y==1,2), 80, 'g', 'filled')

xlabel('x_1')
ylabel('x_2')
title('Clasificación del plano mediante un perceptrón')
axis equal
grid on
hold off