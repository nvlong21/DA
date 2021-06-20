all:
	cd lib/face_detector/retinaface_pytorch/cython/; python setup.py build_ext --inplace; rm -rf build; cd ../../../
clean:
	cd src/face_detector/retinaface_pytorch/cython/; rm *.so *.c *.cpp; cd ../../../

