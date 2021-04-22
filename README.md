Particle-based Viscoelastic Fluid Simulation is a high performance cross-platform simulator for 2D viscoelastic fluid simulation based on the [Taichi](https://github.com/taichi-dev/taichi) programming language.

## Setup
Install taichi on your computer. See https://taichi.readthedocs.io/en/stable/install.html for specification.
Especially, please note that Taichi only supports Python 3.6/3.7/3.8 (64-bit), so please download an appropriate python version if you do not have that.

## Simulate scenes
It will run if you run the python file as usual. It is by default simulating the last scene. To change scene, please go to the line 826 which says create_system8(). Change it to the scene you want. (like to change that line to create_system3()). There are 7 scenes avaliable, 1,2,3,5,6,7,8.
