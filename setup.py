from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = (
    "Census income experiments with classification, clustering, "
    "saved CSV outputs, and notebook-based visualization."
)


def read_requirements(path):
    lines = (ROOT / path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


BASE_REQUIREMENTS = read_requirements("requirements/base.txt")
NOTEBOOK_REQUIREMENTS = read_requirements("requirements/notebook.txt")


setup(
    name="census-cluster",
    version="0.1.0",
    description="Training and visualization pipeline for the census take-home project",
    long_description=README,
    long_description_content_type="text/plain",
    packages=find_packages(include=["model", "model.*", "cluster_tabular_models", "cluster_tabular_models.*"]),
    python_requires=">=3.10,<3.14",
    install_requires=BASE_REQUIREMENTS,
    extras_require={
        "notebook": NOTEBOOK_REQUIREMENTS,
        "full": NOTEBOOK_REQUIREMENTS,
    },
    package_data={
        "cluster_tabular_models": ["configs/*.json"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "census-train=model.train:main",
            "census-foundation-train=cluster_tabular_models.scripts.train_census_foundation_models:main",
        ]
    },
)
