## Generating synthetic data for PVNet

### Prepare the dataset

Download the LINEMOD

```
wget -O LINEMOD.tar --no-check-certificate "https://onedrive.live.com/download?cid=05750EBEE1537631&resid=5750EBEE1537631%21135&authkey=AJRHFmZbcjXxTmI"
```

Download the LINEMOD_ORIG, which can be found at [here](http://campar.in.tum.de/Main/StefanHinterstoisser).

Download the SUN397

```
wget http://groups.csail.mit.edu/vision/SUN/releases/SUN2012pascalformat.tar.gz
```

### Create the soft link

Commands

```
ln -s path/to/LINEMOD ./data/LINEMOD
ln -s path/to/LINEMOD_ORIG ./data/LINEMOD_ORIG
ln -s path/to/SUN2012 ./data/SUN
```

### Configure the blender

Download blender-2.79a, and revise the `cfg.BLENDER_PATH` in the `config.py`.

### Run

**All the Commands must be carried out in the ROOT directory.**

10000 images from blender rendering

```
python run.py --type rendering
```

10000 images from fusion

```
python run.py --type fuse
```

### Examples

Blender rendering

![blender](./assets/blender.png)

Fusion

![fusion](./assets/fuse.png)

