#!/Users/mb312/dev_trees/myscripter/venv/bin/python
# EASY-INSTALL-ENTRY-SCRIPT: 'myscripter==1.0','console_scripts','my_console_script'

#with open('mavis/__init__.py') as f:
#    version = f.readlines()[0].split("=")[1].strip().replace('"', "")

import mavis

__requires__ = f'mavis-core=={mavis.__version__}'
import sys

from pkg_resources import load_entry_point

sys.exit(
    load_entry_point(__requires__, 'console_scripts', 'mavis')()
)