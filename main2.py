import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

# ==========================================
# 1. Настройка интерфейса командной строки (CLI)
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Модель лесного пожара (Клеточный автомат)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры запуска из терминала:
  python LR1.py --size 150 --density 0.7 --steps 300
  python LR1.py -s 100 -d 0.6 -n von_neumann --animate
        """
    )
    parser.add_argument("-s", "--size", type=int, default=100, help="Размер поля (size x size), по умолчанию: 100")
    parser.add_argument("-d", "--density", type=float, default=0.6, help="Начальная плотность деревьев (0.0 - 1.0), по умолчанию: 0.6")
    parser.add_argument("-t", "--steps", type=int, default=200, help="Количество шагов симуляции, по умолчанию: 200")
    parser.add_argument("-p", type=float, default=0.01, help="Вероятность роста дерева, по умолчанию: 0.01")
    parser.add_argument("-f", type=float, default=0.0005, help="Вероятность удара молнии, по умолчанию: 0.0005")
    parser.add_argument("-k", type=int, default=1, help="Порог горящих соседей для возгорания, по умолчанию: 1")
    parser.add_argument("-n", "--neighborhood", type=str, default="moore", 
                        choices=["moore", "von_neumann"], 
                        help="Тип окрестности: 'moore' (8 соседей) или 'von_neumann' (4 соседа)")
    parser.add_argument("--no-plot", action="store_true", help="Не показывать график статистики (ускоряет работу)")
    
    return parser.parse_args()


# ==========================================
# 2. Константы и цветовая карта
# ==========================================
EMPTY = 0   # Пустая клетка / сгоревшее дерево (чёрный)
TREE = 1    # Зелёное дерево (зелёный)
BURNING = 2 # Горящее дерево (красный/оранжевый)

# Цветовая карта: 0 - чёрный, 1 - зелёный, 2 - оранжево-красный
fire_cmap = ListedColormap(['#1a1a1a', '#2d8a2d', '#ff4500'])

# Глобальная настройка шрифта для технических отчетов
plt.rcParams['font.family'] = 'Times New Roman'


# ==========================================
# 3. Логика клеточного автомата
# ==========================================
def create_forest(size: tuple, tree_density: float = 0.4) -> np.ndarray:
    rows, cols = size
    forest = np.zeros((rows, cols), dtype=int)
    random_values = np.random.random((rows, cols))
    forest[random_values < tree_density] = TREE
    return forest


def count_burning_neighbors(forest: np.ndarray, neighborhood: str = 'moore') -> np.ndarray:
    burning_mask = (forest == BURNING).astype(int)
    count = np.zeros_like(forest, dtype=int)
    
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if neighborhood == 'moore':
        shifts += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
    for di, dj in shifts:
        shifted = np.roll(burning_mask, shift=(di, dj), axis=(0, 1))
        # Реализация фиксированных границ (за пределами поля = 0)
        if di == -1: shifted[-1, :] = 0
        elif di == 1: shifted[0, :] = 0
        if dj == -1: shifted[:, -1] = 0
        elif dj == 1: shifted[:, 0] = 0
        count += shifted
        
    return count


def update_forest(forest: np.ndarray, p: float, f: float, k: int = 1, neighborhood: str = 'moore') -> np.ndarray:
    new_forest = forest.copy()
    burning_neighbors = count_burning_neighbors(forest, neighborhood)
    
    is_empty = (forest == EMPTY)
    is_tree = (forest == TREE)
    is_burning = (forest == BURNING)
    
    # Правило 1: Горящее дерево → пустая клетка
    new_forest[is_burning] = EMPTY
    
    # Правило 2: Дерево загорается (от соседей или от молнии)
    ignites_by_neighbor = is_tree & (burning_neighbors >= k)
    lightning = is_tree & (np.random.random(forest.shape) < f)
    new_forest[ignites_by_neighbor | lightning] = BURNING
    
    # Правило 3: Рост дерева на пустой клетке (если нет горящих соседей)
    can_grow = is_empty & (burning_neighbors == 0)
    grows = can_grow & (np.random.random(forest.shape) < p)
    new_forest[grows] = TREE
    
    return new_forest


def simulate_forest_fire(initial_forest: np.ndarray, steps: int, p: float, f: float, k: int = 1, neighborhood: str = 'moore') -> list:
    history = [initial_forest.copy()]
    current_forest = initial_forest
    for _ in range(steps):
        current_forest = update_forest(current_forest, p, f, k, neighborhood)
        history.append(current_forest.copy())
    return history


# ==========================================
# 4. Визуализация
# ==========================================
def plot_forest_states(history: list, title: str = "Модель лесного пожара", steps_to_show: int = 9):
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
            
            n_empty = np.sum(history[step] == EMPTY)
            n_tree = np.sum(history[step] == TREE)
            n_burning = np.sum(history[step] == BURNING)
            
            # Эмодзи удалены, чтобы избежать UserWarning о missing glyphs
            ax.set_title(f'Шаг {step}\nДеревья: {n_tree} | Горят: {n_burning} | Пусто: {n_empty}', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_forest_animation(history: list, filename: str = 'forest_fire.gif', interval: int = 100):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame):
        ax.clear()
        # origin='lower' делает ось Y "человеческой" (0 снизу, растет вверх)
        ax.imshow(history[frame], cmap=fire_cmap, interpolation='nearest', origin='lower')
        ax.set_aspect('equal') 
        
        ax.set_xlabel('X (координата столбца)', fontsize=11)
        ax.set_ylabel('Y (координата строки)', fontsize=11)
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(10)
        
        n_tree = np.sum(history[frame] == TREE)
        n_burning = np.sum(history[frame] == BURNING)
        n_empty = np.sum(history[frame] == EMPTY)
        total = history[frame].size
        
        ax.set_title(f'Время (шаг): {frame}\n'
                     f'Деревья: {n_tree} ({100*n_tree/total:.1f}%) | '
                     f'Горят: {n_burning} ({100*n_burning/total:.1f}%) | '
                     f'Пусто: {n_empty} ({100*n_empty/total:.1f}%)',
                     fontsize=12, fontweight='bold')
        return ax,
    
    anim = FuncAnimation(fig, update, frames=len(history), interval=interval, blit=False)
    
    try:
        anim.save(filename, writer='pillow', fps=10, dpi=150)
        print(f"✅ Анимация успешно сохранена в {filename}")
    except Exception as e:
        print(f"⚠️ Не удалось сохранить анимацию (возможно, не установлен pillow): {e}")
        plt.show()
    plt.close()


def plot_statistics(history: list):
    steps = len(history)
    n_trees = [np.sum(state == TREE) for state in history]
    n_burning = [np.sum(state == BURNING) for state in history]
    n_empty = [np.sum(state == EMPTY) for state in history]
    
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
# 5. Точка входа (Запуск программы)
# ==========================================
if __name__ == "__main__":
    # 1. Получаем параметры из CLI (или используем default, если запущено через кнопку Run)
    args = parse_arguments()
    
    SIZE = (args.size, args.size)
    TREE_DENSITY = args.density
    STEPS = args.steps
    p = args.p
    f = args.f
    k = args.k
    NEIGHBORHOOD = args.neighborhood
    
    print("="*50)
    print("🌲 ЗАПУСК МОДЕЛИ ЛЕСНОГО ПОЖАРА")
    print("="*50)
    print(f"Размер поля      : {SIZE[0]} x {SIZE[1]}")
    print(f"Плотность леса   : {TREE_DENSITY}")
    print(f"Количество шагов : {STEPS}")
    print(f"Параметры (p, f, k): {p}, {f}, {k}")
    print(f"Окрестность      : {NEIGHBORHOOD}")
    print("="*50)

    print("Создание начального состояния леса...")
    initial_forest = create_forest(SIZE, tree_density=TREE_DENSITY)
    
    # Гарантируем начало пожара, добавляя 3 очага возгорания
    initial_forest[SIZE[0]//2, SIZE[1]//2] = BURNING
    initial_forest[SIZE[0]//3, SIZE[1]//3] = BURNING
    initial_forest[2*SIZE[0]//3, 2*SIZE[1]//3] = BURNING
    
    print(f"Запуск симуляции на {STEPS} шагов...")
    history = simulate_forest_fire(initial_forest, STEPS, p, f, k, NEIGHBORHOOD)
    
    print("Визуализация результатов...")
    # Раскомментируйте строку ниже, если хотите увидеть сетку кадров
    # plot_forest_states(history, title=f"Модель лесного пожара ({NEIGHBORHOOD})")
    
    if not args.no_plot:
        plot_statistics(history)
    
    create_forest_animation(history, 'forest_fire.gif', interval=50)
    
    print("✅ Симуляция успешно завершена!")