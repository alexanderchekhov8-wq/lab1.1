import numpy as np
import matplotlib
matplotlib.use('Agg')  # Чтобы окна не открывались при массовой генерации
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import os

# ==========================================
# Константы состояний
# ==========================================
EMPTY = 0
TREE = 1
BURNING = 2

fire_cmap = ListedColormap(['#1a1a1a', '#2d8a2d', '#ff4500'])

# Папка для выходных файлов
OUTPUT_DIR = "forest_fire_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Качество картинок
DPI = 200


# ==========================================
# Генерация начального леса
# ==========================================
def create_forest(size, tree_density=0.4, seed=None):
    if seed is not None:
        np.random.seed(seed)
    rows, cols = size
    forest = np.zeros((rows, cols), dtype=int)
    random_values = np.random.random((rows, cols))
    forest[random_values < tree_density] = TREE
    return forest


# ==========================================
# Подсчёт горящих соседей (ТОЛЬКО фон Нейман)
# ==========================================
def count_burning_neighbors(forest):
    """
    Подсчёт горящих соседей в окрестности фон Неймана (4 соседа):
    сверху, снизу, слева, справа. Диагонали не учитываются.
    """
    burning_mask = (forest == BURNING).astype(int)
    count = np.zeros_like(forest, dtype=int)

    # Только 4 ортогональных направления
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for di, dj in offsets:
        shifted = np.roll(burning_mask, shift=(di, dj), axis=(0, 1))
        # Фиксированные границы: обнуляем завёрнутые края
        if di == -1: shifted[-1, :] = 0
        elif di == 1: shifted[0, :] = 0
        if dj == -1: shifted[:, -1] = 0
        elif dj == 1: shifted[:, 0] = 0
        count += shifted

    return count


# ==========================================
# Один такт модели
# ==========================================
def update_forest(forest, p, f, k=1):
    """
    Один такт модели лесного пожара.
    Используется окрестность фон Неймана (4 соседа).
    """
    new_forest = forest.copy()
    burning_neighbors = count_burning_neighbors(forest)

    is_empty = (forest == EMPTY)
    is_tree = (forest == TREE)
    is_burning = (forest == BURNING)

    # Правило 1: горящее дерево → пусто
    new_forest[is_burning] = EMPTY

    # Правило 2: возгорание
    ignites_by_neighbor = is_tree & (burning_neighbors >= k)
    lightning = is_tree & (np.random.random(forest.shape) < f)
    new_forest[ignites_by_neighbor | lightning] = BURNING

    # Правило 3: рост нового дерева
    can_grow = is_empty & (burning_neighbors == 0)
    grows = can_grow & (np.random.random(forest.shape) < p)
    new_forest[grows] = TREE

    return new_forest


# ==========================================
# Симуляция
# ==========================================
def simulate_forest_fire(initial_forest, steps, p, f, k=1):
    history = [initial_forest.copy()]
    current = initial_forest
    for _ in range(steps):
        current = update_forest(current, p, f, k)
        history.append(current.copy())
    return history


# ==========================================
# Сохранение 4 ключевых кадров (2x2)
# ==========================================
def save_states_image(history, title, filename, steps_to_show=4):
    n_plots = min(steps_to_show, len(history))
    indices = np.linspace(0, len(history) - 1, n_plots, dtype=int)

    cols = 2
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12), dpi=DPI)
    axes = np.array(axes).flatten()

    for idx, ax in enumerate(axes):
        if idx < n_plots:
            step = indices[idx]
            ax.imshow(history[step], cmap=fire_cmap, interpolation='nearest')

            n_empty = int(np.sum(history[step] == EMPTY))
            n_tree = int(np.sum(history[step] == TREE))
            n_burning = int(np.sum(history[step] == BURNING))

            ax.set_title(f'Шаг {step}\n'
                         f'Деревья: {n_tree}  |  Горят: {n_burning}  |  Пусто: {n_empty}',
                         fontsize=13, fontweight='bold', pad=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Сохранено изображение: {filename}")


# ==========================================
# Сохранение графика статистики
# ==========================================
def save_statistics_image(history, title, filename):
    steps = len(history)
    n_trees, n_burning, n_empty = [], [], []
    for state in history:
        n_trees.append(int(np.sum(state == TREE)))
        n_burning.append(int(np.sum(state == BURNING)))
        n_empty.append(int(np.sum(state == EMPTY)))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)
    time = np.arange(steps)
    ax.plot(time, n_trees, label='Деревья', color='#2d8a2d', linewidth=2.5)
    ax.plot(time, n_burning, label='Горят', color='#ff4500', linewidth=2.5)
    ax.plot(time, n_empty, label='Пусто', color='#1a1a1a', linewidth=2.5)
    ax.set_xlabel('Время (шаг)', fontsize=14)
    ax.set_ylabel('Количество клеток', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Сохранён график: {filename}")


# ==========================================
# Сохранение GIF-анимации
# ==========================================
def save_animation(history, title, filename, interval=100):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    def update(frame):
        ax.clear()
        ax.imshow(history[frame], cmap=fire_cmap, interpolation='nearest')
        n_tree = int(np.sum(history[frame] == TREE))
        n_burning = int(np.sum(history[frame] == BURNING))
        n_empty = int(np.sum(history[frame] == EMPTY))
        total = history[frame].size
        ax.set_title(f'{title}\nВремя: {frame}\n'
                     f'Деревья: {n_tree} ({100*n_tree/total:.1f}%) | '
                     f'Горят: {n_burning} ({100*n_burning/total:.1f}%) | '
                     f'Пусто: {n_empty} ({100*n_empty/total:.1f}%)',
                     fontsize=11)
        ax.axis('off')
        return ax,

    anim = FuncAnimation(fig, update, frames=len(history),
                         interval=interval, blit=False)
    anim.save(filename, writer='pillow', fps=10)
    plt.close()
    print(f"  Сохранена анимация: {filename}")


# ==========================================
# Общая функция для запуска сценария
# ==========================================
def run_scenario(name, description, initial_forest, steps, p, f, k):
    print("\n" + "=" * 70)
    print(f"  СЦЕНАРИЙ: {name}")
    print(f"  {description}")
    print(f"  Параметры: p={p}, f={f}, k={k}, окрестность=фон Неймана, шагов={steps}")
    print("=" * 70)

    history = simulate_forest_fire(initial_forest, steps, p, f, k)

    safe_name = name.lower().replace(" ", "_").replace(".", "")
    base = os.path.join(OUTPUT_DIR, safe_name)

    save_states_image(history, f"{name}: 4 ключевых кадра", f"{base}_states.png")
    save_statistics_image(history, f"{name}: динамика", f"{base}_stats.png")
    save_animation(history, name, f"{base}_animation.gif")

    final = history[-1]
    print(f"  Финальное состояние: "
          f"деревья={int(np.sum(final == TREE))}, "
          f"горят={int(np.sum(final == BURNING))}, "
          f"пусто={int(np.sum(final == EMPTY))}")


# ==========================================
# СЦЕНАРИЙ 1. Случайный лес с очагами возгорания
# ==========================================
def scenario_1_random_with_fire():
    size = (100, 100)
    forest = create_forest(size, tree_density=0.5, seed=42)
    forest[size[0]//2, size[1]//2] = BURNING
    forest[size[0]//3, size[1]//3] = BURNING
    forest[2*size[0]//3, 2*size[1]//3] = BURNING
    run_scenario(
        "1. Случайный лес с очагами возгорания",
        "Начальная плотность 50%, три очага возгорания.",
        forest, steps=200, p=0.01, f=0.0005, k=1
    )


# ==========================================
# СЦЕНАРИЙ 2. Высокая плотность деревьев
# ==========================================
def scenario_2_high_density():
    size = (100, 100)
    forest = create_forest(size, tree_density=0.8, seed=1)
    forest[size[0]//2, size[1]//2] = BURNING
    run_scenario(
        "2. Высокая плотность деревьев (80%)",
        "Плотный лес — огонь распространяется очень быстро.",
        forest, steps=150, p=0.01, f=0.0005, k=1
    )


# ==========================================
# СЦЕНАРИЙ 3. Низкая плотность деревьев
# ==========================================
def scenario_3_low_density():
    size = (100, 100)
    forest = create_forest(size, tree_density=0.2, seed=2)
    forest[size[0]//2, size[1]//2] = BURNING
    run_scenario(
        "3. Низкая плотность деревьев (20%)",
        "Разреженный лес — огонь часто затухает сам.",
        forest, steps=200, p=0.01, f=0.0005, k=1
    )


# ==========================================
# СЦЕНАРИЙ 4. Быстрый рост деревьев
# ==========================================
def scenario_4_growth_probability():
    size = (100, 100)
    forest = create_forest(size, tree_density=0.5, seed=3)
    forest[size[0]//2, size[1]//2] = BURNING
    run_scenario(
        "4. Быстрый рост деревьев (p=0.05)",
        "Лес восстанавливается быстро, пожары локальны.",
        forest, steps=200, p=0.05, f=0.0005, k=1
    )


# ==========================================
# СЦЕНАРИЙ 5. Удары молнии
# ==========================================
def scenario_5_lightning():
    size = (100, 100)
    forest = create_forest(size, tree_density=0.5, seed=4)
    forest[size[0]//2, size[1]//2] = BURNING
    run_scenario(
        "5. Удары молнии (f=0.001)",
        "Пожары возникают случайно по всему полю.",
        forest, steps=200, p=0.01, f=0.001, k=1
    )


# ==========================================
# СЦЕНАРИЙ 6. Медленный рост деревьев
# ==========================================
def scenario_6_slow_growth():
    size = (100, 100)
    forest = create_forest(size, tree_density=0.5, seed=5)
    forest[size[0]//2, size[1]//2] = BURNING
    run_scenario(
        "6. Медленный рост деревьев (p=0.001)",
        "Лес восстанавливается медленно — долго остаётся пустым.",
        forest, steps=200, p=0.001, f=0.0005, k=1
    )


# ==========================================
# ЗАПУСК ВСЕХ 6 СЦЕНАРИЕВ
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("  ЗАПУСК 6 СЦЕНАРИЕВ МОДЕЛИ ЛЕСНОГО ПОЖАРА")
    print("  Окрестность: фон Неймана (4 соседа)")
    print(f"  Все результаты будут сохранены в папку: {OUTPUT_DIR}")
    print(f"  DPI картинок: {DPI} (высокое качество)")
    print("=" * 70)

    scenario_1_random_with_fire()
    scenario_2_high_density()
    scenario_3_low_density()
    scenario_4_growth_probability()
    scenario_5_lightning()
    scenario_6_slow_growth()

    print("\n" + "=" * 70)
    print("  ВСЕ 6 СЦЕНАРИЕВ ЗАВЕРШЕНЫ")
    print(f"  Файлы сохранены в: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)
    print("\nДля каждого сценария создано три файла:")
    print("  *_states.png      — 4 ключевых кадра (сетка 2x2)")
    print("  *_stats.png       — график динамики деревьев/огня/пустоты")
    print("  *_animation.gif   — полная анимация сценария")