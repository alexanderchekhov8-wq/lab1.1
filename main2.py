import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Модель лесного пожара")
    parser.add_argument("--size", type=int, default=100, help="Размер поля (size x size)")
    parser.add_argument("--density", type=float, default=0.6, help="Начальная плотность деревьев (0.0 - 1.0)")
    parser.add_argument("--steps", type=int, default=200, help="Количество шагов симуляции")
    parser.add_argument("--p", type=float, default=0.01, help="Вероятность роста дерева")
    parser.add_argument("--f", type=float, default=0.0005, help="Вероятность молнии")
    parser.add_argument("--k", type=int, default=1, help="Порог горящих соседей")
    
    # ДОБАВЛЕН ВВОД ОКРЕСТНОСТИ:
    parser.add_argument("--neighborhood", type=str, default="moore", 
                        choices=["moore", "von_neumann"], 
                        help="Тип окрестности: 'moore' (8 соседей) или 'von_neumann' (4 соседа)")
    
    return parser.parse_args()




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

# ==========================================
# Константы состояний
# ==========================================
EMPTY = 0   # Пустая клетка / сгоревшее дерево (чёрный)
TREE = 1    # Зелёное дерево (зелёный)
BURNING = 2 # Горящее дерево (красный/оранжевый)

# Цветовая карта: 0 - чёрный, 1 - зелёный, 2 - оранжево-красный
fire_cmap = ListedColormap(['#1a1a1a', '#2d8a2d', '#ff4500'])


def create_forest(size: tuple, tree_density: float = 0.4) -> np.ndarray:
    """
    Создаёт начальное состояние леса.
    Случайное распределение деревьев, горящих нет.
    
    Args:
        size: (rows, cols) - размер поля
        tree_density: доля клеток, занятых деревьями (0 до 1)
    """
    rows, cols = size
    forest = np.zeros((rows, cols), dtype=int)
    
    # Случайно размещаем деревья
    random_values = np.random.random((rows, cols))
    forest[random_values < tree_density] = TREE
    
    # Можно добавить несколько очагов возгорания вручную
    # forest[rows//2, cols//2] = BURNING
    
    return forest


def count_burning_neighbors(forest: np.ndarray, neighborhood: str = 'moore') -> np.ndarray:
    """
    Подсчитывает количество горящих соседей для каждой клетки.
    
    Args:
        forest: текущее состояние леса
        neighborhood: 'moore' (8 соседей) или 'von_neumann' (4 соседа)
    
    Returns:
        Массив с количеством горящих соседей для каждой клетки
    """
    # Создаём маску горящих клеток
    burning_mask = (forest == BURNING).astype(int)
    
    if neighborhood == 'moore':
        # Окрестность Мура (8 соседей)
        count = np.zeros_like(forest, dtype=int)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                # Фиксированные границы: за пределами поля считаем 0 (нет деревьев)
                shifted = np.roll(burning_mask, shift=(di, dj), axis=(0, 1))
                # Обнуляем края, чтобы реализовать фиксированные границы
                if di == -1:
                    shifted[-1, :] = 0
                elif di == 1:
                    shifted[0, :] = 0
                if dj == -1:
                    shifted[:, -1] = 0
                elif dj == 1:
                    shifted[:, 0] = 0
                count += shifted
    elif neighborhood == 'von_neumann':
        # Окрестность фон Неймана (4 соседа)
        count = np.zeros_like(forest, dtype=int)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(burning_mask, shift=(di, dj), axis=(0, 1))
            if di == -1:
                shifted[-1, :] = 0
            elif di == 1:
                shifted[0, :] = 0
            if dj == -1:
                shifted[:, -1] = 0
            elif dj == 1:
                shifted[:, 0] = 0
            count += shifted
    
    return count


def update_forest(forest: np.ndarray, p: float, f: float, k: int = 1, 
                  neighborhood: str = 'moore') -> np.ndarray:
    """
    Обновляет состояние леса на один шаг (синхронное обновление).
    
    Правила (из лекции и слайда):
    1. Сгоревшее дерево (2) становится пустой клеткой (0)
    2. Зелёное дерево (1) становится горящим (2), если:
       - в окрестности >= k горящих деревьев, ИЛИ
       - с вероятностью f (удар молнии)
    3. На пустой клетке (0) вырастает дерево (1) с вероятностью p,
       если в окрестности нет ни одного горящего
    
    Args:
        forest: текущее состояние леса
        p: вероятность роста нового дерева на пустой клетке
        f: вероятность удара молнии (самовозгорания дерева)
        k: порог горящих соседей для возгорания
        neighborhood: тип окрестности
    
    Returns:
        Новое состояние леса
    """
    new_forest = forest.copy()
    
    # Подсчитываем горящих соседей
    burning_neighbors = count_burning_neighbors(forest, neighborhood)
    
    # Маски для каждого состояния
    is_empty = (forest == EMPTY)
    is_tree = (forest == TREE)
    is_burning = (forest == BURNING)
    
    # Правило 1: Горящее дерево → пустая клетка
    new_forest[is_burning] = EMPTY
    
    # Правило 2: Дерево загорается
    # 2a. Если рядом >= k горящих соседей
    ignites_by_neighbor = is_tree & (burning_neighbors >= k)
    # 2b. Молния (случайное возгорание)
    lightning = is_tree & (np.random.random(forest.shape) < f)
    # Объединяем оба условия
    new_forest[ignites_by_neighbor | lightning] = BURNING
    
    # Правило 3: Рост дерева на пустой клетке
    # Только если нет горящих соседей
    can_grow = is_empty & (burning_neighbors == 0)
    # С вероятностью p
    grows = can_grow & (np.random.random(forest.shape) < p)
    new_forest[grows] = TREE
    
    return new_forest


def simulate_forest_fire(initial_forest: np.ndarray, steps: int, 
                         p: float, f: float, k: int = 1,
                         neighborhood: str = 'moore') -> list:
    """
    Запускает симуляцию лесного пожара.
    
    Returns:
        history: список состояний леса на каждом шаге
    """
    history = [initial_forest.copy()]
    current_forest = initial_forest
    
    for step in range(steps):
        current_forest = update_forest(current_forest, p, f, k, neighborhood)
        history.append(current_forest.copy())
    
    return history


def plot_forest_states(history: list, title: str = "Модель лесного пожара",
                       steps_to_show: int = 9):
    """
    Визуализирует несколько ключевых кадров эволюции леса.
    """
    n_plots = min(steps_to_show, len(history))
    indices = np.linspace(0, len(history) - 1, n_plots, dtype=int)
    
    cols = 3
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()
    
    for idx, ax in enumerate(axes):
        if idx < n_plots:
            step = indices[idx]
            ax.imshow(history[step], cmap=fire_cmap, interpolation='nearest')
            
            # Подсчёт статистики
            n_empty = np.sum(history[step] == EMPTY)
            n_tree = np.sum(history[step] == TREE)
            n_burning = np.sum(history[step] == BURNING)
            
            ax.set_title(f'Шаг {step}\n'
                        f'🌲 Деревья: {n_tree} |  Горят: {n_burning} | ⬛ Пусто: {n_empty}',
                        fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_forest_animation(history: list, filename: str = 'forest_fire.gif',
                            interval: int = 100):
    """
    Создаёт анимацию лесного пожара с координатными осями.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Устанавливаем шрифт Times New Roman для всех элементов осей
    plt.rcParams['font.family'] = 'Times New Roman'
    
    def update(frame):
        ax.clear()
        # Добавляем origin='lower', чтобы 0 был внизу оси Y
        ax.imshow(history[frame], cmap=fire_cmap, interpolation='nearest', origin='lower')
        
        # Включаем оси и делаем клетки строго квадратными
        ax.set_aspect('equal') 
        
        # Добавляем подписи осей
        ax.set_xlabel('X ', fontsize=11)
        ax.set_ylabel('Y ', fontsize=11)
        
        # Настраиваем шрифт и размер меток на осях (тиков)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(10)
        
        # Статистика
        n_tree = np.sum(history[frame] == TREE)
        n_burning = np.sum(history[frame] == BURNING)
        n_empty = np.sum(history[frame] == EMPTY)
        total = history[frame].size
        
        # Заголовок с текущим шагом и статистикой
        ax.set_title(f'Время (шаг): {frame}\n'
                     f'Деревья: {n_tree} ({100*n_tree/total:.1f}%) | '
                     f'Горят: {n_burning} ({100*n_burning/total:.1f}%) | '
                     f'Пусто: {n_empty} ({100*n_empty/total:.1f}%)',
                     fontsize=12, fontweight='bold')
        
        return ax,
    
    anim = FuncAnimation(fig, update, frames=len(history), interval=interval, blit=False)
    
    try:
        # Сохранение с высоким DPI для четкости
        anim.save(filename, writer='pillow', fps=10, dpi=150)
        print(f"✅ Анимация успешно сохранена в {filename}")
    except Exception as e:
        print(f"⚠️ Не удалось сохранить анимацию: {e}")
        plt.show()
    plt.close()

def plot_statistics(history: list):
    """
    Строит графики изменения количества деревьев, горящих и пустых клеток во времени.
    """
    steps = len(history)
    n_trees = []
    n_burning = []
    n_empty = []
    
    for state in history:
        n_trees.append(np.sum(state == TREE))
        n_burning.append(np.sum(state == BURNING))
        n_empty.append(np.sum(state == EMPTY))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    time = np.arange(steps)
    
    ax.plot(time, n_trees, label='Деревья', color='#2d8a2d', linewidth=2)
    ax.plot(time, n_burning, label='Горят', color='#ff4500', linewidth=2)
    ax.plot(time, n_empty, label='Пусто', color='#1a1a1a', linewidth=2)
    
    ax.set_xlabel('Время (шаг)', fontsize=12)
    ax.set_ylabel('Количество клеток', fontsize=12)
    ax.set_title('Динамика лесного пожара', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ==========================================
# Пример запуска
# ==========================================

if __name__ == "__main__":
    args = parse_arguments()
    
    SIZE = (args.size, args.size)
    TREE_DENSITY = args.density
    STEPS = args.steps
    p = args.p
    f = args.f
    k = args.k
    NEIGHBORHOOD = args.neighborhood  # <-- Забираем значение отсюда
    
    print(f" Запуск с параметрами: размер={SIZE[0]}, плотность={TREE_DENSITY}, окрестность={NEIGHBORHOOD}")

    print("🌲 Создание начального состояния леса...")
    initial_forest = create_forest(SIZE, tree_density=TREE_DENSITY)
    
    # Добавим несколько очагов возгорания вручную
    # (иначе пожар не начнётся, если f очень маленькое)
    initial_forest[SIZE[0]//2, SIZE[1]//2] = BURNING
    initial_forest[SIZE[0]//3, SIZE[1]//3] = BURNING
    initial_forest[2*SIZE[0]//3, 2*SIZE[1]//3] = BURNING
    
    print(f"🔥 Запуск симуляции лесного пожара на {STEPS} шагов...")
    print(f"   Параметры: p={p}, f={f}, k={k}, окрестность={NEIGHBORHOOD}")
    
    history = simulate_forest_fire(initial_forest, STEPS, p, f, k, NEIGHBORHOOD)
    
    print("📊 Визуализация результатов...")
    # plot_forest_states(history, title="Модель лесного пожара (окрестность Мура)")
    
    # График статистики
    plot_statistics(history)
    
    # Создание анимации (раскомментируйте, если нужно)
    create_forest_animation(history, 'forest_fire.gif', interval=50)
    
    print("✅ Симуляция завершена!")