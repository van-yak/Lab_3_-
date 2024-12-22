import numpy as np
import time

def bfgs(f, grad_f, x0, tol=1e-6, max_iter=100):
    """
    Реализация алгоритма BFGS для минимизации функции f.
    """
    x = x0
    n = len(x0)
    H = np.eye(n)  # Начальная приближенная гессианая матрица
    start_time = time.time()

    for i in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)

        # Проверка критерия остановки
        if grad_norm < tol:
            break

        # Направление поиска
        p = -H @ grad

        # Линейный поиск
        alpha = line_search(f, grad_f, x, p)
        if alpha is None or alpha <= 0:
            print("Линейный поиск не нашел подходящего шага. Остановка.")
            break

        # Обновление x
        x_new = x + alpha * p

        # Обновление градиента и разности
        s = x_new - x
        y = grad_f(x_new) - grad

        # Проверка на корректность обновления
        if np.dot(y, s) <= 1e-10:
            print("Плохая аппроксимация гессиана. Остановка.")
            break

        rho = 1.0 / np.dot(y, s)
        I = np.eye(n)

        # Обновление H по формуле BFGS
        H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x = x_new

    elapsed_time = time.time() - start_time
    return x, f(x), i, elapsed_time

def line_search(f, grad_f, x, p, alpha=1.0, beta=0.5, c=1e-4):
    """
    Простая реализация линейного поиска с условием Армихо.
    """
    while f(x + alpha * p) > f(x) + c * alpha * grad_f(x).T @ p:
        alpha *= beta
        if alpha < 1e-8:  # Предотвращение слишком малого шага
            return None
    return alpha

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

    x_min, f_min, n_iter, elapsed_time = bfgs(f, grad_f, x0)
    print(f"Минимум найден в точке: {x_min}")
    print(f"Значение функции в минимуме: {f_min}")
    print(f"Количество итераций: {n_iter}")
    print(f"Время работы: {elapsed_time:.6f} секунд")