"""
generator.py

Responsible for creating the project structure.
"""

from pathlib import Path

from rich.console import Console

console = Console()


def create_project(project: dict, recommendation: dict) -> None:
    """
    Create the project structure.

    Parameters
    ----------
    project : dict
        User project information.

    recommendation : dict
        Recommended template information.
    """

    project_name = project["name"]
    project_path = Path(project["path"])
    project_directory = project_path / project_name

    # Check if project already exists
    if project_directory.exists():
        console.print(
            f"[bold red]Error:[/bold red] Project '{project_name}' already exists."
        )
        return

    console.print("\n[cyan]Generating project...[/cyan]")

    # Create main project directory
    project_directory.mkdir(parents=True)

    # Create folders
    (project_directory / "src").mkdir()
    (project_directory / "tests").mkdir()

    # Create files
    (project_directory / "README.md").touch()
    (project_directory / "requirements.txt").touch()
    (project_directory / "pyproject.toml").touch()

    console.print(
        f"[bold green]✓ Project '{project_name}' created successfully![/bold green]"
    )

    console.print(f"\nLocation: {project_directory}")