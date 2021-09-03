#!/Users/mb312/dev_trees/myscripter/venv/bin/python
# EASY-INSTALL-ENTRY-SCRIPT: 'myscripter==1.0','console_scripts','my_console_script'
__requires__ = 'mavis==0.1.0'
import sys

from pkg_resources import load_entry_point

sys.exit(
load_entry_point('mavis==0.1.0', 'console_scripts', 'mavis')()
)