"""
main.py — Головна точка входу (Команда 10)

Запуск:
    python src/main.py          # інтерактивне меню
    python src/main.py --task 1 # одразу запустити завдання 1
    python src/main.py --task 2 -- --q-min -5 --q-max 5   # з параметрами

© 2026 Команда 10 — Фрактальні методи аналізу
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║      Фрактальні методи аналізу сигналів та зображень        ║
║                     © 2026 Команда 10                        ║
╠══════════════════════════════════════════════════════════════╣
║  [1] Завдання 1   — ARMA(1,1) та ARIMA(2,1,2) (EUR/USD)    ║
║  [2] Завдання 2   — ARFIMA + MF-DFA                         ║
║  [3] Завдання 3.10 — Фрактальна розмірність зображень       ║
║  [4] Завдання 4.11 — IMU фрактальний аналіз (UCI HAR)       ║
║  [0] Вихід                                                   ║
╚══════════════════════════════════════════════════════════════╝"""

HINTS = {
    '1': (
        "  Параметри (додаються після --):                         \n"
        "    --ticker   тікер yfinance      (default: EURUSD=X)   \n"
        "    --start    дата початку        (default: 2015-01-01) \n"
        "    --end      дата кінця          (default: 2024-12-31) \n"
        "    --test-ratio  частка тесту     (default: 0.2)        "
    ),
    '2': (
        "  Параметри (додаються після --):                         \n"
        "    --q-min    мін. момент MF-DFA  (default: -3)         \n"
        "    --q-max    макс. момент MF-DFA (default: 3)          \n"
        "    --train-ratio  частка train    (default: 0.8)        "
    ),
    '3': (
        "  Параметри (додаються після --):                         \n"
        "    --size        розмір зображення px  (default: 512)   \n"
        "    --noise-sigma рівень шуму σ          (default: auto) "
    ),
    '4': (
        "  Параметри (додаються після --):                         \n"
        "    --n-windows   вікон UCI HAR на сегмент (default: 12) \n"
        "    --threshold   поріг H нестабільності   (default: 0.5)\n"
        "    --window-size розмір ковзного вікна     (default: 200)"
    ),
}

TASKS = {
    '1': 'task1_timeseries',
    '2': 'task2_fractal_ts',
    '3': 'task3_image_fractal',
    '4': 'task4_imu',
}


def run_task(choice: str, extra_args: list):
    module_name = TASKS[choice]
    print(f"\n{'─' * 64}")
    print(f"  Підказка — доступні параметри:\n{HINTS[choice]}")
    print(f"{'─' * 64}\n")

    import importlib
    mod = importlib.import_module(module_name)

    # Override sys.argv so the task's argparse sees any extra args passed here
    sys.argv = [module_name + '.py'] + extra_args
    mod.main()


def interactive_menu():
    print(BANNER)
    while True:
        try:
            choice = input("\nОберіть завдання [0–4]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВихід.")
            break

        if choice == '0':
            print("До побачення!")
            break
        elif choice in TASKS:
            print(f"\n  Додаткові аргументи (Enter — без змін): ", end='')
            try:
                raw = input().strip()
            except (EOFError, KeyboardInterrupt):
                raw = ''
            extra = raw.split() if raw else []
            run_task(choice, extra)
        else:
            print("  Невірний вибір. Введіть число від 0 до 4.")


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="© 2026 Команда 10 — Головна точка входу."
    )
    p.add_argument('--task', choices=['1', '2', '3', '4'], default=None,
                   help='Запустити конкретне завдання без меню')
    args, extra = p.parse_known_args()

    if args.task:
        print(BANNER)
        run_task(args.task, extra)
    else:
        interactive_menu()


if __name__ == '__main__':
    main()
