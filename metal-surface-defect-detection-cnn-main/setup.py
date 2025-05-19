from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str)->List[str]:

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

    if "-e ." in requirements:
        requirements.remove("-e .")

    return requirements

setup(
    name = 'Metal_Surface_Defect_Detection',
    version = '0.0.1',
    author = 'Pradeep',
    author_email = 'xyz@gmail.com',
    packages = find_packages(),
    install_requries = get_requirements('requirements.txt')
)