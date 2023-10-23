import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='rxhands-unam-colab',  
     version='0.18',
     author="Arturo Curiel",
     author_email="me@arturocuriel.com",
     description="Labeling of finger landmarks in hand xrays.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/elmugrearturo/rxhands",
     packages=['rxhands'],
     package_dir={'rxhands' : 'src/rxhands'},
     package_data={
         "rxhands" : ["bin/*.bin"],
     },
     install_requires=['wheel', 'numpy', 'matplotlib', 'opencv-python', 'pandas', 'scikit-image', 'scikit-learn', 'tensorflow'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     entry_points={
         'console_scripts' : [
             'rxhands=rxhands.main:main'
             ]
         }
 )
