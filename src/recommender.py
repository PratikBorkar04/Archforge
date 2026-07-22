"""
recommender.py

Responsible for selecting the appropriate project template.
"""


def get_template(project):
    """
    Return the template based on project type.

    Parameters
    ----------
    project : dict

    Returns
    -------
    dict
    """

    project_type = project["type"]

    templates = {
        "ml": "ml",
        "cv": "cv",
        "nlp": "nlp",
    }

    return {
        "template": templates[project_type]
    }