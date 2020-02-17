from setuptools import setup


setup(name='gym_object_manipulation',
      version=1.0,
      description='OpenAI Gym Object Manipulation, modified from the OpenAI robotics Mujoco fetch environments',
      zip_safe=False,
      install_requires=[
          'numpy', 'gym', 'mujoco_py>=1.50' 'imageio'
      ],
      package_data={'gym': [
        'assets/LICENSE.md',
        'assets/fetch/*.xml',
        'assets/stls/fetch/*.stl',
        'assets/textures/*.png']
      }
)
