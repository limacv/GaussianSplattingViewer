# Tiny Gaussian Splatting Viewer
![UI demo](assets/teaser.png)
This is a simple Gaussian Splatting Viewer built with PyOpenGL. It's easy to install with minimum dependencies. The goal of this project is to provide a minimum example of the viewer for research and study purpose. 


# Usage
Install the dependencies:
```
pip install -r requirements.txt
```

Launch the viewer:
```
python main.py
```

Check how to use UI in the "help" panel.

The Gaussian file loader is compatiable with the official implementation. 
Therefore, download pretrained Gaussian PLY file from [this official link](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip), and select the "point_cloud.ply" you like by clicking the 'open ply' button, and you are all set!


# Troubleshoot

The rendering speed of is comparable to the official CUDA renderer. If you're experiencing slow rendering, it's likely you are using integrated graphics card instead of a high-performance one. You can configure python to use high-performance graphics card in system settings. In Windows, you can set in Setting > System > Display > Graphics. See the screenshot below for example.

![Setting > System > Display > Graphics](assets/setting.png)

# Limitations
- The implementation utilizes SSBO, which is only support by OpenGL version >= 4.3. Although this version is widely adopted, MacOS is an exception. As a result, this viewer does not support MacOS.

- Currently, the sorting of the Gaussians uses numpy `argsort`, which is not very efficient. So there is a button to manually toggle sorting. However it's interesting to see what it looks like when the gaussian is wrongly sorted.


# TODO
- Tighter billboard to reduce number of fragments
- Benchmark the rendering w/ official implementation.
- Better sorting implementation.
- Save viewing parameters
