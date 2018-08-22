all:
	clang++ -g -I/home/gornet/Documents/APP2/vaa3d_tools/released_plugins/v3d_plugins/neurontracing_vn2/app2/basic_c_fun \
		-I/home/gornet/Documents/APP2/vaa3d_tools/released_plugins/v3d_plugins/neurontracing_vn2/app2 \
		-ltiff \
		-L/home/gornet/Documents/APP2/vaa3d_tools/released_plugins/v3d_plugins/neurontracing_vn2/app2/libtiff/tiff-4.0.9/libtiff \
		/home/gornet/Documents/APP2/vaa3d_tools/released_plugins/v3d_plugins/neurontracing_vn2/app2/basic_c_fun/stackutil.cpp \
		/home/gornet/Documents/APP2/vaa3d_tools/released_plugins/v3d_plugins/neurontracing_vn2/app2/basic_c_fun/mg_utilities.cpp \
		/home/gornet/Documents/APP2/vaa3d_tools/released_plugins/v3d_plugins/neurontracing_vn2/app2/basic_c_fun/mg_image_lib.cpp \
		*.cpp -o app2

# all:
# 	clang++ -g -I/home/gornet/Documents/APP2/v3d_external/v3d_main/basic_c_fun \
		-I/home/gornet/Documents/APP2/v3d_external/v3d_main/common_lib/include \
		-I/usr/include/x86_64-linux-gnu -lv3dtiff \
		-L/home/gornet/Documents/APP2/v3d_external/v3d_main/common_lib/lib \
		-L/home/gornet/Documents/APP2/vaa3d_tools/released_plugins/v3d_plugins/neurontracing_vn2/app2/libtiff/tiff-4.0.9/build/libtiff \
		/home/gornet/Documents/APP2/v3d_external/v3d_main/basic_c_fun/stackutil.cpp \
		/home/gornet/Documents/APP2/v3d_external/v3d_main/basic_c_fun/mg_utilities.cpp \
		/home/gornet/Documents/APP2/v3d_external/v3d_main/basic_c_fun/mg_image_lib.cpp \
		*.cpp -o app2

# all:
# 	clang++ -g -O3 -Wall -shared -std=c++11 -fPIC  \
# 		-I/home/gornet/Documents/APP2/v3d_external/v3d_main/basic_c_fun \
# 		-I/home/gornet/Documents/APP2/v3d_external/v3d_main/common_lib/include \
# 		-I/usr/include/x86_64-linux-gnu -lv3dtiff \
# 		-L/home/gornet/Documents/APP2/v3d_external/v3d_main/common_lib/lib \
# 		`python3 -m pybind11 --includes` \
# 		/home/gornet/Documents/APP2/v3d_external/v3d_main/basic_c_fun/stackutil.cpp \
# 		/home/gornet/Documents/APP2/v3d_external/v3d_main/basic_c_fun/mg_utilities.cpp \
# 		/home/gornet/Documents/APP2/v3d_external/v3d_main/basic_c_fun/mg_image_lib.cpp \
# 		*.cpp -o app2`python3-config --extension-suffix`
