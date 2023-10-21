# Simple Gaussian Splatting Viewer
This is a simple PyOpenGL based Gaussian Splatting Viewer. 
It's easy to install with minimum dependencies. 
The goal of this project is to provide a minimum example of the viewer for research and study purpose.

# Usage
Install the dependencies:
```
pip install -r requirements.txt
```

Start viewer:
```
python main.py
```

Check how to use UI in the "help" panel

# Limitations
- The implementation depends on SSBO, which is only support by OpenGL version >= 4.3. This is a widely support version except for MacOS. Therefore, runing on MacOS is not supported.

- Currently the Gaussian sorting uses numpy argsort, which is not very efficient. 

# TODO
- Better camera controls
- Window size change callback
- Tighter billboard to reduce number of fragments
- Benchmark the rendering w/ official implementation
- Better sorting algorithm