WORK_DIR=~/work

virualenv $WORK_DIR
cd ~$WORK_DIR
source bin/activate
pip install -U pip
pip install https://pypi.python.org/packages/14/fb/85915ac0004a34fc6a027ba0b742a86bdc501d16349be5525c773cd16a56/opencv_python-3.1.0.3-cp27-cp27mu-manylinux1_x86_64.whl#md5=aadbe88cce72260b8dae8fc872f94568
pip install youtube_dl
pip install pytube