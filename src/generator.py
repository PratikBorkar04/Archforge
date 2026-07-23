"""
generator.py

Responsible for generating projects from templates.
"""

from pathlib import Path
import shutil

from rich.console import Console

console = Console()


def create_project(project: dict, recommendation: dict) -> None:
    """
    Generate a project by copying the selected template.
    """

    project_name = project["name"]
    project_location = Path(project["path"])

    destination = project_location / project_name

    if destination.exists():
        console.print(
            f"[bold red]Error:[/bold red] Project '{project_name}' already exists."
        )
        return

    template_name = recommendation["template"]

    template_path = (
        Path(__file__).parent
        / "templates"
        / template_name
    )

    if not template_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Template '{template_name}' not found."
        )
        return

    console.print("\n[cyan]Generating project...[/cyan]")

    shutil.copytree(template_path, destination)

    console.print(
        f"\n[bold green]✓ Project '{project_name}' created successfully![/bold green]"
    )

    console.print(f"Location : {destination}")