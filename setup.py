from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'visualnav_transformer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('deployment/launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('deployment/config/*.yaml')),
        (os.path.join('share', package_name, 'params'), glob('deployment/params/*.yaml')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='divyamc',
    maintainer_email='divyamchandalia3@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'navigate = deployment.src.navigate:main',
            'explore = deployment.src.explore:main',
            'pd_controller = deployment.src.pd_controller:main',
            'create_topomap = deployment.src.create_topomap:main',
        ],
    },
)
