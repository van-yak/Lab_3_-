import numpy as np
import time

def gradient_descent(f, grad_f, x0, learning_rate=0.1, tol=1e-6, max_iter=1000):
    """
    Реализация метода градиентного спуска для минимизации функции f.

    Параметры:
        f (callable): Целевая функция.
        grad_f (callable): Градиент целевой функции.
        x0 (np.array): Начальная точка.
        learning_rate (float): Шаг обучения (скорость спуска).
        tol (float): Точность для остановки.
        max_iter (int): Максимальное количество итераций.

    Возвращает:
        x (np.array): Найденная точка минимума.
        f_val (float): Значение функции в точке минимума.
        n_iter (int): Количество итераций.
        elapsed_time (float): Время работы алгоритма в секундах.
    """
    x = x0
    start_time = time.time()

    for i in range(max_iter):
        grad = grad_f(x)

        # Проверка критерия остановки
        if np.linalg.norm(grad) < tol:
            break

        # Обновление точки
        x = x - learning_rate * grad

    elapsed_time = time.time() - start_time
    return x, f(x), i, elapsed_time

# Пример использования
if __name__ == "__main__":
    # Пример: минимизация квадратичной функции f(x) = 0.5 * x^T * A * x - b^T * x
    A = np.array([[3, 2], [2, 6]])
    b = np.array([2, 3])

    def f(x):
        return 0.5 * x.T @ A @ x - b.T @ x

    def grad_f(x):
        return A @ x - b

    x0 = np.array([0.0, 0.0])
    learning_rate = 0.1

    x_min, f_min, n_iter, elapsed_time = gradient_descent(f, grad_f, x0, learning_rate)
    print(f"Минимум найден в точке: {x_min}")
    print(f"Значение функции в минимуме: {f_min}")
    print(f"Количество итераций: {n_iter}")
    print(f"Время работы: {elapsed_time:.6f} секунд")