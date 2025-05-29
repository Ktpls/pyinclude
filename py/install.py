import os
filedir=os.path.dirname(__file__)
os.system(rf'echo "export PYTHONPATH=\$PYTHONPATH:{filedir}" >> ~/.bashrc')