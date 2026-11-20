
"""
app.py

Main entry point for ArchForge.
Supports Manual and Description-Based generation.
"""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.recommender import get_template
from src.generator import create_project
from src.description_mode.description_analyzer import analyze_description

console = Console()


# ============================================================
# UI
# ============================================================

def display_banner():
    console.print(
        Panel.fit(
            "[bold cyan]ArchForge[/bold cyan]\n"
            "AI Project Architecture Generator",
            border_style="cyan",
        )
    )


def display_shortcuts():
    console.print(
        "\n[dim]Shortcuts: [H] Home   [B] Back   [Q] Quit[/dim]"
    )


def handle_shortcut(value):
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
        console.print("[red]Invalid project type selected.[/red]")
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
    console.print("\n[bold cyan]Description Based Mode[/bold cyan]")
    display_shortcuts()

    description = input("\nProject Description: ").strip()

    shortcut = handle_shortcut(description)
    if shortcut:
        return shortcut

    if not description:
        console.print("[red]Project description cannot be empty.[/red]")
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
    console.print("\n[bold green]Project Summary[/bold green]")

    if "name" in project:
        console.print(f"Name        : {project['name']}")

    if "type" in project:
        console.print(f"Type        : {project['type'].upper()}")

    if "description" in project:
        console.print(f"Description : {project['description']}")

    console.print(f"Path        : {project['path']}")


# ============================================================
# Description Architecture Preview
# ============================================================

def display_architecture(architecture):
    console.print(
        "\n[bold green]Detected Project Architecture[/bold green]"
    )

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Component", style="cyan", min_width=20)
    table.add_column("Detected Configuration")

    table.add_row(
        "Project Name",
        str(architecture.get("project_name", "Not identified"))
    )

    table.add_row(
        "Domain",
        str(architecture.get("domain", "Not identified")).upper()
    )

    table.add_row(
        "Task",
        str(architecture.get("task", "Not identified"))
    )

    table.add_row(
        "Algorithm Selection",
        str(architecture.get("algorithm_mode", "Not identified"))
    )

    algorithms = architecture.get("algorithms", [])

    algorithm_names = [
        algorithm.get("name", algorithm.get("id", "Unknown"))
        for algorithm in algorithms
    ]

    table.add_row(
        "Algorithms",
        ", ".join(algorithm_names) if algorithm_names else "None"
    )

    table.add_row(
        "Preprocessing",
        ", ".join(architecture.get("preprocessing", [])) or "None"
    )

    table.add_row(
        "Representations",
        ", ".join(architecture.get("representations", [])) or "None"
    )

    table.add_row(
        "Evaluation Metrics",
        ", ".join(architecture.get("evaluation_metrics", [])) or "None"
    )

    table.add_row(
        "Dataset",
        str(architecture.get("dataset", {}).get("name") or "Not specified")
    )

    table.add_row(
        "Interface",
        str(architecture.get("interface", "none")).upper()
    )

    console.print(table)


# ============================================================
# Description Analysis
# ============================================================

def run_description_analysis(project):
    console.print(
        "\n[bold cyan]Analyzing Project Description...[/bold cyan]"
    )

    try:
        result = analyze_description(project["description"])

    except (FileNotFoundError, ValueError) as error:
        console.print(f"\n[bold red]Analyzer Error:[/bold red] {error}")
        return

    except Exception as error:
        console.print(
            f"\n[bold red]Unexpected Analysis Error:[/bold red] {error}"
        )
        return

    if not result.get("success"):
        console.print(
            "\n[bold red]Could not generate a valid architecture.[/bold red]"
        )

        if result.get("message"):
            console.print(result["message"])

        for error in result.get("errors", []):
            console.print(f"[red]- {error}[/red]")

        return

    architecture = result["architecture"]

    display_architecture(architecture)

    validation = architecture.get("validation", {})

    if not validation.get("is_valid", False):
        console.print(
            "\n[bold red]Architecture validation failed.[/bold red]"
        )

        for error in validation.get("errors", []):
            console.print(f"[red]- {error}[/red]")

        return

    console.print(
        "\n[bold green]Description analysis completed successfully![/bold green]"
    )

    console.print(
        "\n[yellow]Architecture is ready for generator integration.[/yellow]"
    )

    # The architecture must be adapted to the existing
    # generator's expected input format before calling
    # create_project(). We will connect that after verifying
    # generator.py to avoid breaking Manual Mode.


# ============================================================
# Main Application
# ============================================================

def main():
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
        # Manual Mode - Existing Pipeline Unchanged
        # ----------------------------------------------------

        if generation_mode == "manual":

            project = get_project_info()

            if isinstance(project, str):
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

            input("\nPress Enter to return to Home...")
            continue

        # ----------------------------------------------------
        # Description-Based Mode
        # ----------------------------------------------------

        if generation_mode == "description":

            project = get_description_info()

            if isinstance(project, str):
                if project == "quit":
                    console.print(
                        "\n[dim]ArchForge closed successfully.[/dim]"
                    )
                    break

                if project in ("home", "back"):
                    continue

            print_summary({
                "description": project["description"],
                "path": project["path"],
            })

            run_description_analysis(project)

            input("\nPress Enter to return to Home...")
            continue


if __name__ == "__main__":
    main()
