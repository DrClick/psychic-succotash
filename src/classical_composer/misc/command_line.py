# flake8: noqa: W605
"""
Provides functionality working with the command line.

Includes:
- Generate Intro for the CLI
"""


def generate_intro() -> None:
    """Generate an ASCII art intro for the CLI."""
    intro_text = """\n

     ____  _____ _              \033[0;32m ____       _       _ _   _       \033[0m
    / ___||  ___| |             \033[0;32m|  _ \  ___| | ___ (_) |_| |_ ___ \033[0m
    \___ \| |_  | |      _____  \033[0;32m| | | |/ _ \ |/ _ \| | __| __/ _ \\\033[0m
     ___) |  _| | |___  |_____| \033[0;32m| |_| |  __/ | (_) | | |_| ||  __/\033[0m
    |____/|_|   |_____|         \033[0;32m|____/ \___|_|\___/|_|\__|\__\___|\033[0m
    
    
    Welcome to the Classical Composer pipeline!

    You can incorporate this module into your own projects or use as standalone pipeline.

    Please see the  README.md for more information on how to use this pipeline/repo.

    """
    print(intro_text)
