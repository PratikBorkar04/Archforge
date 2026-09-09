"""
app.py

Main entry point for ArchForge.
"""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from src.recommender import get_template
from src.generator import create_project

console = Console()


# ============================================================
# UI
# ============================================================

def display_banner():
    """Display welcome banner."""

    console.print(
        Panel.fit(
            "[bold cyan]ArchForge[/bold cyan]\n"
            "AI Project Architecture Generator",
            border_style="cyan",
        )
    )


def display_shortcuts():
    """Display available navigation shortcuts."""

    console.print(
        "\n[dim]Shortcuts: [H] Home   [B] Back   [Q] Quit[/dim]"
    )


def handle_shortcut(value):
    """
    Check whether the input is a navigation shortcut.

    Returns
    -------
    str or None
        Shortcut name if detected.
    """

    value = value.strip().lower()

    if value == "h":
        return "home"

    if value == "b":
        return "back"

    if value == "q":
        return "quit"

    return None


# ============================================================
# Generation Mode
# ============================================================

def select_generation_mode():
    """Select project generation mode."""

    while True:

        console.print("\n[bold cyan]Select Generation Mode[/bold cyan]")

        display_shortcuts()

        console.print("\n1. Manual")
        console.print("2. Description Based")

        choice = input("\nEnter choice (1-2): ").strip()

        shortcut = handle_shortcut(choice)

        if shortcut:
            return shortcut

        if choice == "1":
            return "manual"

        if choice == "2":
            return "description"

        console.print(
            "[bold red]Invalid choice. Please enter 1 or 2.[/bold red]"
        )


# ============================================================
# Manual Mode
# ============================================================

def get_project_info():
    """
    Collect project information for manual generation.

    Returns
    -------
    dict or str
        Project configuration or navigation command.
    """

    console.print("\n[bold cyan]Manual Mode[/bold cyan]")

    display_shortcuts()

    project_name = input("\nProject Name: ").strip()

    shortcut = handle_shortcut(project_name)

    if shortcut:
        return shortcut

    if not project_name:
        console.print("[red]Project name cannot be empty.[/red]")
        return get_project_info()

    console.print("\nSelect Project Type")

    console.print("1. Machine Learning")
    console.print("2. Natural Language Processing")

    choice = input("\nEnter choice (1-2): ").strip()

    shortcut = handle_shortcut(choice)

    if shortcut:
        return shortcut

    project_types = {
        "1": "ml",
        "2": "nlp",
    }

    project_type = project_types.get(choice)

    if project_type is None:
        console.print(
            "[bold red]Invalid project type selected.[/bold red]"
        )
        return get_project_info()

    location = input(
        "\nProject Location (Press Enter for current directory): "
    ).strip()

    shortcut = handle_shortcut(location)

    if shortcut:
        return shortcut

    if not location:
        location = str(Path.cwd())

    return {
        "name": project_name,
        "type": project_type,
        "path": location,
    }


# ============================================================
# Description Mode
# ============================================================

def get_description_info():
    """
    Collect project information from a description.

    Returns
    -------
    dict or str
        Description configuration or navigation command.
    """

    console.print("\n[bold cyan]Description Based Mode[/bold cyan]")

    display_shortcuts()

    description = input("\nProject Description: ").strip()

    shortcut = handle_shortcut(description)

    if shortcut:
        return shortcut

    if not description:
        console.print(
            "[red]Project description cannot be empty.[/red]"
        )
        return get_description_info()

    location = input(
        "\nProject Location (Press Enter for current directory): "
    ).strip()

    shortcut = handle_shortcut(location)

    if shortcut:
        return shortcut

    if not location:
        location = str(Path.cwd())

    return {
        "description": description,
        "path": location,
    }


# ============================================================
# Summary
# ============================================================

def print_summary(project):
    """Display project summary."""

    console.print("\n[bold green]Project Summary[/bold green]")

    if "name" in project:
        console.print(f"Name : {project['name']}")

    if "type" in project:
        console.print(f"Type : {project['type'].upper()}")

    if "description" in project:
        console.print(f"Description : {project['description']}")

    console.print(f"Path : {project['path']}")


# ============================================================
# Main Application
# ============================================================

def main():
    """Main application loop."""

    while True:

        display_banner()

        generation_mode = select_generation_mode()

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if generation_mode == "quit":
            console.print(
                "\n[dim]ArchForge closed successfully.[/dim]"
            )
            break

        # ----------------------------------------------------
        # Home
        # ----------------------------------------------------

        if generation_mode == "home":
            continue

        # ----------------------------------------------------
        # Manual Mode
        # ----------------------------------------------------

        if generation_mode == "manual":

            project = get_project_info()

            if project == "quit":
                console.print(
                    "\n[dim]ArchForge closed successfully.[/dim]"
                )
                break

            if project in ("home", "back"):
                continue

            print_summary(project)

            recommendation = get_template(project)

            create_project(project, recommendation)

            console.print(
                "\n[bold green]Project created successfully![/bold green]"
            )

            input(
                "\nPress Enter to return to Home..."
            )

            continue

        # ----------------------------------------------------
        # Description Based Mode
        # ----------------------------------------------------

        if generation_mode == "description":

            project = get_description_info()

            if project == "quit":
                console.print(
                    "\n[dim]ArchForge closed successfully.[/dim]"
                )
                break

            if project in ("home", "back"):
                continue

            print_summary(project)

            console.print(
                "\n[yellow]Description-based generation "
                "is currently under development.[/yellow]"
            )

            input(
                "\nPress Enter to return to Home..."
            )

            continue


if __name__ == "__main__":
    main()
