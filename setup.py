from setuptools import find_packages, setup
import os



HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> list[str]:
    '''
    This function fetches and parse the
    requirements.txt content into a list.
    '''

    requirements=[]

    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        # [req.replace("\n","") for req in requirements]
        requirements = [req.strip() for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements



setup(

    name= 'ML_Project_1',
    version='0.1',
    author='Hmmmmmmmmmmmmmmmmmmmmmmm',
    author_email='aftabaqueelkhan@gmail.com',
    packages=find_packages(),
    # install_requires=get_requirements('requirements.txt'),
    install_requires=get_requirements(os.path.join(os.path.dirname(__file__), 'requirements.txt')),

)