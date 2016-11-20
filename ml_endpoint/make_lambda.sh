wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.9/opencv-2.4.9.zip
mkdir opencv_install
unzip opencv-2.4.9.zip -d opencv_install/
cd opencv_install/
cmake -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/local/ -D PYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python -D PYTHON_PACKAGES_PATH=$VIRTUAL_ENV/lib/python2.7/site-packages -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_SHARED_LIBS=NO -D BUILD_NEW_PYTHON_SUPPORT=ON -D BUILD_opencv_gpu=OFF -D BUILD_DOCS=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF 

WORK_DIR=~/work

virualenv $WORK_DIR
cd ~$WORK_DIR
source bin/activate
pip install -U pip
#pip install https://pypi.python.org/packages/14/fb/85915ac0004a34fc6a027ba0b742a86bdc501d16349be5525c773cd16a56/opencv_python-3.1.0.3-cp27-cp27mu-manylinux1_x86_64.whl#md5=aadbe88cce72260b8dae8fc872f94568
pip install youtube_dl
pip install pytube