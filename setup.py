from setuptools import setup
from setuptools.dist import Distribution

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        _python, _abi, plat = _bdist_wheel.get_tag(self)
        return "py3", "none", plat


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": bdist_wheel})
