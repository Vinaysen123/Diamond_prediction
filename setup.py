from setuptools import find_packages , setup
from typing import List

val = '-e .'

def get_reuirements(file_path : str)->List[str]:
    requirements = [] 
    with open(file_path) as file_obj:
       requirements = file_obj.readlines()
       requirements = [req.replace("\n","") for req in requirements]

       if val in requirements:
           requirements.remove(val)  

       return requirements 


setup(
    name = "Diamond price" ,
    version = "0.0.1" ,
    author = "Vinay" , 
    author_email= "vinaysenvidisha@gmail.com",
    install_requires = get_reuirements('requirement.txt'),
    packages= find_packages() 

)