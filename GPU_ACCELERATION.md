```bash
sudo apt-get update
```

```bash
sudo apt-get gcc pkg-config zlib1g-dev libexpat1-dev libexrandr-dev
```

```bash
./configure --prefix=/usr/local                                   \
            --enable-opengl --disable-gles1 --disable-gles2   \
            --disable-va --disable-xvmc --disable-vdpau       \
            --enable-shared-glapi                             \
            --enable-gallium-llvm --enable-llvm-shared-libs   \
            --with-gallium-drivers=swrast,swr                 \
            --disable-glx                                     \
            --disable-osmesa --enable-gallium-osmesa          \
            ac_cv_path_LLVM_CONFIG=llvm-config-6.0
```

```bash
make -j8
```
```bashS
sudo make install
```