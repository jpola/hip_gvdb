hipcc --shaered -o libgvdb.so ../tests/main_rendertofile.cpp -I./ -I../../shared_assets -I ../../sample_utils  gvdb_cutils.cpp app_perf.cpp gvdb_allocator.cpp gvdb_camera.cpp gvdb_model.cpp gvdb_node.cpp gvdb_render_opengl.cpp gvdb_scene.cpp gvdb_types.cpp gvdb_vec.cpp gvdb_volume_3D.cpp gvdb_volume_base.cpp gvdb_volume_gvdb.cpp loader_ObjarReader.cpp loader_OBJReader.cpp loader_Parser.cpp string_helper.cpp -I /opt/rocm/hip/include/ -I ../kernels/ -I /opt/rocm/rocprim/include/ -I /opt/rocm/rocthrust/include/  -std=c++14 -lGLEW -lGL -fPIC 