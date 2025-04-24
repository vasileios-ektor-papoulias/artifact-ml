import importlib
import os
import pkgutil

package_dir = os.path.dirname(__file__)
package_name = __name__
for _, module_name, _ in pkgutil.walk_packages([package_dir], package_name + "."):
    importlib.import_module(module_name)
