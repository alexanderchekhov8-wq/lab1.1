import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Чтобы окна не открывались при массовой генерации
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import os

# ==========================================
# 1. Настройка интерфейса командной строки (CLI)
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Модель лесного пожара: запуск сценариев с гибкой настройкой.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры запуска:
  python script.py --scenarios 1 3 5          # Запустить только сценарии 1, 3 и 5
  python script.py --scenarios all --dpi 150  # Запустить все сценарии с DPI 150 (быстрее)
  python script.py -s 2 --override-steps 300  # Сценарий 2, но продлить до 300 шагов
  python script.py -s 4 --override-p 0.1      # Сценарий 4, но с очень быстрым ростом леса
        """
    )
    parser.add_argument("-s", "--scenarios", nargs="+", default=["all"],
                        help="Номера сценариев для запуска (1-6) или 'all' для всех. По умолчанию: all")
    parser.add_argument("-o", "--output-dir", type=str, default="forest_fire_output",
                        help="Папка для сохранения результатов. По умолчанию: forest_fire_output")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Разрешение сохраняемых PNG-изображений. По умолчанию: 200")
    
    # Переопределения параметров (позволяют исследовать сценарии с новыми значениями)
    parser.add_argument("--override-steps", type=int, default=None, help="Переопределить количество шагов")
    parser.add_argument("--override-p", type=float, default=None, help="Переопределить вероятность роста дерева (p)")
    parser.add_argument("--override-f", type=float, default=None, help="Переопределить вероятность молнии (f)")
    parser.add_argument("--override-k", type=int, default=None, help="Переопределить порог возгорания (k)")
    
    return parser.parse_args()


# ==========================================
# 2. Константы и конфигурация сценариев
# ==========================================
EMPTY = 0
TREE = 1
BURNING = 2

fire_cmap = ListedColormap(['#1a1a1a', '#2d8a2d', '#ff4500'])

# Конфигурация всех 6 сценариев в одном месте для удобства управления
SCENARIOS = {
    1: {"name": "1. Случайный лес с очагами", "desc": "Плотность 50%, три очага возгорания.", "density": 0.5, "seed": 42, "steps": 200, "p": 0.01, "f": 0.0005, "k": 1, "ignitions": "three"},
    2: {"name": "2. Высокая плотность (80%)", "desc": "Плотный лес — огонь распространяется очень быстро.", "density": 0.8, "seed": 1, "steps": 150, "p": 0.01, "f": 0.0005, "k": 1, "ignitions": "one"},
    3: {"name": "3. Низкая плотность (20%)", "desc": "Разреженный лес — огонь часто затухает сам.", "density": 0.2, "seed": 2, "steps": 200, "p": 0.01, "f": 0.0005, "k": 1, "ignitions": "one"},
    4: {"name": "4. Быстрый рост (p=0.05)", "desc": "Лес восстанавливается быстро, пожары локальны.", "density": 0.5, "seed": 3, "steps": 200, "p": 0.05, "f": 0.0005, "k": 1, "ignitions": "one"},
    5: {"name": "5. Удары молнии (f=0.001)", "desc": "Пожары возникают случайно по всему полю.", "density": 0.5, "seed": 4, "steps": 200, "p": 0.01, "f": 0.001, "k": 1, "ignitions": "one"},
    6: {"name": "6. Медленный рост (p=0.001)", "desc": "Лес восстанавливается медленно — долго остаётся пустым.", "density": 0.5, "seed": 5, "steps": 200, "p": 0.001, "f": 0.0005, "k": 1, "ignitions": "one"},
}


# ==========================================
# 3. Логика клеточного автомата
# ==========================================
def create_forest(size, tree_density=0.4, seed=None):
    if seed is not None:
        np.random.seed(seed)
    rows, cols = size
    forest = np.zeros((rows, cols), dtype=int)
    random_values = np.random.random((rows, cols))
    forest[random_values < tree_density] = TREE
    return forest


def count_burning_neighbors(forest):
    """Подсчёт горящих соседей в окрестности фон Неймана (4 соседа)."""
    burning_mask = (forest == BURNING).astype(int)
    count = np.zeros_like(forest, dtype=int)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for di, dj in offsets:
        shifted = np.roll(burning_mask, shift=(di, dj), axis=(0, 1))
        if di == -1: shifted[-1, :] = 0
        elif di == 1: shifted[0, :] = 0
        if dj == -1: shifted[:, -1] = 0
        elif dj == 1: shifted[:, 0] = 0
        count += shifted
    return count


def update_forest(forest, p, f, k=1):
    new_forest = forest.copy()
    burning_neighbors = count_burning_neighbors(forest)

    is_empty = (forest == EMPTY)
    is_tree = (forest == TREE)
    is_burning = (forest == BURNING)

    new_forest[is_burning] = EMPTY
    ignites_by_neighbor = is_tree & (burning_neighbors >= k)
    lightning = is_tree & (np.random.random(forest.shape) < f)
    new_forest[ignites_by_neighbor | lightning] = BURNING

    can_grow = is_empty & (burning_neighbors == 0)
    grows = can_grow & (np.random.random(forest.shape) < p)
    new_forest[grows] = TREE

    return new_forest


def simulate_forest_fire(initial_forest, steps, p, f, k=1):
    history = [initial_forest.copy()]
    current = initial_forest
    for _ in range(steps):
        current = update_forest(current, p, f, k)
        history.append(current.copy())
    return history


# ==========================================
# 4. Визуализация и сохранение
# ==========================================
def save_states_image(history, title, filename, steps_to_show=4, dpi=200):
    n_plots = min(steps_to_show, len(history))
    indices = np.linspace(0, len(history) - 1, n_plots, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=dpi)
    axes = np.array(axes).flatten()

    for idx, ax in enumerate(axes):
        if idx < n_plots:
            step = indices[idx]
            ax.imshow(history[step], cmap=fire_cmap, interpolation='nearest')
            n_empty = int(np.sum(history[step] == EMPTY))
            n_tree = int(np.sum(history[step] == TREE))
            n_burning = int(np.sum(history[step] == BURNING))

            ax.set_title(f'Шаг {step}\nДеревья: {n_tree} | Горят: {n_burning} | Пусто: {n_empty}',
                         fontsize=13, fontweight='bold', pad=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Сохранено изображение: {os.path.basename(filename)}")


def save_statistics_image(history, title, filename, dpi=200):
    steps = len(history)
    n_trees, n_burning, n_empty = [], [], []
    for state in history:
        n_trees.append(int(np.sum(state == TREE)))
        n_burning.append(int(np.sum(state == BURNING)))
        n_empty.append(int(np.sum(state == EMPTY)))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
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
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Сохранён график: {os.path.basename(filename)}")


def save_animation(history, title, filename, interval=100, dpi=100):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    plt.rcParams['font.family'] = 'Times New Roman' # Ваш предпочтительный шрифт

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

        n_tree = int(np.sum(history[frame] == TREE))
        n_burning = int(np.sum(history[frame] == BURNING))
        n_empty = int(np.sum(history[frame] == EMPTY))
        total = history[frame].size
        
        ax.set_title(f'{title}\nВремя: {frame}\n'
                     f'Деревья: {n_tree} ({100*n_tree/total:.1f}%) | '
                     f'Горят: {n_burning} ({100*n_burning/total:.1f}%) | '
                     f'Пусто: {n_empty} ({100*n_empty/total:.1f}%)',
                     fontsize=11)
        return ax,

    anim = FuncAnimation(fig, update, frames=len(history), interval=interval, blit=False)
    anim.save(filename, writer='pillow', fps=10, dpi=dpi)
    plt.close()
    print(f"  Сохранена анимация: {os.path.basename(filename)}")


# ==========================================
# 5. Исполняющая логика сценариев
# ==========================================
def run_scenario_from_config(scenario_id, config, args):
    name = config["name"]
    desc = config["desc"]

    # Применяем переопределения из CLI, если они были переданы
    steps = args.override_steps if args.override_steps is not None else config["steps"]
    p = args.override_p if args.override_p is not None else config["p"]
    f = args.override_f if args.override_f is not None else config["f"]
    k = args.override_k if args.override_k is not None else config["k"]

    print("\n" + "=" * 70)
    print(f"  СЦЕНАРИЙ {scenario_id}: {name}")
    print(f"  {desc}")
    print(f"  Параметры: p={p}, f={f}, k={k}, шаги={steps}, окрестность=фон Неймана")
    print("=" * 70)

    size = (100, 100)
    forest = create_forest(size, tree_density=config["density"], seed=config["seed"])

    # Расстановка очагов возгорания в зависимости от сценария
    if config["ignitions"] == "three":
        forest[size[0]//2, size[1]//2] = BURNING
        forest[size[0]//3, size[1]//3] = BURNING
        forest[2*size[0]//3, 2*size[1]//3] = BURNING
    else:
        forest[size[0]//2, size[1]//2] = BURNING

    history = simulate_forest_fire(forest, steps, p, f, k)

    # Формируем безопасное имя файла
    safe_name = name.lower().replace(" ", "_").replace(".", "").replace("(", "").replace(")", "").replace("=", "")
    base = os.path.join(args.output_dir, f"scenario_{scenario_id}_{safe_name}")

    save_states_image(history, f"{name}: 4 ключевых кадра", f"{base}_states.png", dpi=args.dpi)
    save_statistics_image(history, f"{name}: динамика", f"{base}_stats.png", dpi=args.dpi)
    save_animation(history, name, f"{base}_animation.gif", dpi=100) # Для GIF dpi=100 оптимально для размера файла

    final = history[-1]
    print(f"  Финальное состояние: "
          f"деревья={int(np.sum(final == TREE))}, "
          f"горят={int(np.sum(final == BURNING))}, "
          f"пусто={int(np.sum(final == EMPTY))}")


# ==========================================
# 6. Точка входа (Main)
# ==========================================
if __name__ == "__main__":
    args = parse_arguments()
    
    # Определяем, какие сценарии запускать
    if "all" in [str(x).lower() for x in args.scenarios]:
        to_run = [1, 2, 3, 4, 5, 6]
    else:
        try:
            to_run = sorted([int(x) for x in args.scenarios])
        except ValueError:
            print("❌ Ошибка: номера сценариев должны быть целыми числами от 1 до 6 или словом 'all'")
            exit(1)

    # Фильтруем некорректные номера
    to_run = [x for x in to_run if 1 <= x <= 6]
    if not to_run:
        print("❌ Не выбрано ни одного корректного сценария.")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("  ЗАПУСК МОДЕЛИ ЛЕСНОГО ПОЖАРА")
    print(f"  Выбранные сценарии: {', '.join(map(str, to_run))}")
    print(f"  Папка вывода: {os.path.abspath(args.output_dir)}")
    print(f"  DPI изображений: {args.dpi}")
    if args.override_steps or args.override_p or args.override_f or args.override_k:
        print("  ⚠️ Включены переопределения параметров (--override-...)")
    print("=" * 70)

    for scenario_id in to_run:
        run_scenario_from_config(scenario_id, SCENARIOS[scenario_id], args)

    print("\n" + "=" * 70)
    print("  ВСЕ ВЫБРАННЫЕ СЦЕНАРИИ ЗАВЕРШЕНЫ")
    print("=" * 70)