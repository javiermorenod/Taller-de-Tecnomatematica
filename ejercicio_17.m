clc; clear; close all;

% Parámetros
eta = 0.1;
epocas = 200;

% Inicialización
w = zeros(epocas+1,1);   % w(1) = w0 = 0
L = zeros(epocas+1,1);   % pérdida

% Funciones
loss = @(w) (w.^3 - w - 1).^2;
grad = @(w) 2*(w.^3 - w - 1).*(3*w.^2 - 1);

% Entrenamiento
for n = 1:epocas
    L(n) = loss(w(n));
    w(n+1) = w(n) - eta * grad(w(n));
end
L(epocas+1) = loss(w(epocas+1));

% Mostrar primeras épocas (comparables con cálculo a mano)
fprintf('Época    w           L\n');
for i = 1:10
    fprintf('%3d   % .6f   %.6f\n', i-1, w(i), L(i));
end

% Resultado final
fprintf('\nResultado final tras %d épocas:\n', epocas);
fprintf('w ≈ %.6f\n', w(end));
fprintf('L ≈ %.6f\n', L(end));

% Gráfica de convergencia
figure;
plot(0:epocas, L, 'o-');
xlabel('Época');
ylabel('L(w)');
title('Convergencia de la función de pérdida');
grid on;