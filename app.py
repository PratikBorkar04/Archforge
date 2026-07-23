"""
app.py

Main entry point for ArchForge.
Responsible for:
1. Collecting user input.
2. Calling the recommendation engine.
3. Generating the project.
"""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from src.recommender import get_template
from src.generator import create_project
import src.recommender as recommender

console = Console()


def display_banner():
    """Display welcome banner."""

    console.print(
        Panel.fit(
            "[bold cyan]ArchForge[/bold cyan]\n"
            "AI Project Architecture Generator",
            border_style="cyan",
        )
    )


def get_project_info():
    """
    Collect project information from the user.

    Returns
    -------
    dict
        Project configuration.
    """

    console.print()

    project_name = input("Project Name: ").strip()

    console.print("\nSelect Project Type")

    console.print("1. Machine Learning")
    console.print("2. Deep Learning")
    console.print("3. Computer Vision")
    console.print("4. Natural Language Processing")
    console.print("5. Retrieval-Augmented Generation")

    choice = input("\nEnter choice (1-4): ").strip()

    project_types = {
        "1": "ml",
        "2": "dl",
        "3": "cv",
        "4": "nlp",
        "5": "rag",
    }

    project_type = project_types.get(choice)

    if project_type is None:
        raise ValueError("Invalid project type selected.")

    location = input(
        "\nProject Location (Press Enter for current directory): "
    ).strip()

    if not location:
        location = str(Path.cwd())

    return {
        "name": project_name,
        "type": project_type,
        "path": location,
    }


def print_summary(project):
    """Display project summary."""

    console.print("\n[bold green]Project Summary[/bold green]")

    console.print(f"Name : {project['name']}")
    console.print(f"Type : {project['type'].upper()}")
    console.print(f"Path : {project['path']}")


def main():
    """Main application."""

    display_banner()

    project = get_project_info()

    print_summary(project)

    recommendation = get_template(project)

    create_project(project, recommendation)

    console.print(
        "\n[bold green]Project created successfully![/bold green]"
    )


if __name__ == "__main__":
    main()