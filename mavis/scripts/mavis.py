#!/Users/mb312/dev_trees/myscripter/venv/bin/python
# EASY-INSTALL-ENTRY-SCRIPT: 'myscripter==1.0','console_scripts','my_console_script'

#with open('mavis/__init__.py') as f:
#    version = f.readlines()[0].split("=")[1].strip().replace('"', "")

__requires__ = 'mavis-core==0.4.4'
import sys

from pkg_resources import load_entry_point

sys.exit(
    load_entry_point(f'mavis-core==0.4.4', 'console_scripts', 'mavis')()
)