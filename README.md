# CLEVR Dataset Generation New Version

This is the code used to generate the [CLEVR dataset](http://cs.stanford.edu/people/jcjohns/clevr/) as described in the paper:

**[CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](http://cs.stanford.edu/people/jcjohns/clevr/)**
 <br>
 <a href='http://cs.stanford.edu/people/jcjohns/'>Justin Johnson</a>,
 <a href='http://home.bharathh.info/'>Bharath Hariharan</a>,
 <a href='https://lvdmaaten.github.io/'>Laurens van der Maaten</a>,
 <a href='http://vision.stanford.edu/feifeili/'>Fei-Fei Li</a>,
 <a href='http://larryzitnick.org/'>Larry Zitnick</a>,
 <a href='http://www.rossgirshick.info/'>Ross Girshick</a>
 <br>
 Presented at [CVPR 2017](http://cvpr2017.thecvf.com/)

Code and pretrained models for the baselines used in the paper [can be found here](https://github.com/facebookresearch/clevr-iep).

This repository is based on the original CLEVR dataset generation project and has been updated so that image generation can run with Blender 3.6 and CUDA 12.

Main upgrades in this version:

- Image generation supports setting a random seed and controlling the number of generated images.
- Question generation supports setting a random seed, generating only one or more selected question types and adding post prompts.
- Question generation provides an interface for writing Chain-of-Thought (CoT) content for each question type.

You can use this code to render synthetic images and compositional questions for those images, like this:

<div align="center">
  <img src="images/example1080.png" width="800px">
</div>

**Q:** How many small spheres are there? <br>
**A:** 2

**Q:**  What number of cubes are small things or red metal objects? <br>
**A:**  2

**Q:** Does the metal sphere have the same color as the metal cylinder? <br>
**A:** Yes

**Q:** Are there more small cylinders than metal things? <br>
**A:** No

**Q:**  There is a cylinder that is on the right side of the large yellow object behind the blue ball; is there a shiny cube in front of it? <br>
**A:**  Yes

If you find this code useful in your research then please cite

```
@inproceedings{johnson2017clevr,
  title={CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning},
  author={Johnson, Justin and Hariharan, Bharath and van der Maaten, Laurens
          and Fei-Fei, Li and Zitnick, C Lawrence and Girshick, Ross},
  booktitle={CVPR},
  year={2017}
}
```

The original code was developed and tested on OSX and Ubuntu 16.04. This version has been adapted for Blender 3.6, whose bundled Python is 3.10, and can use CUDA 12 compatible NVIDIA drivers for GPU rendering.

## Step 1: Generating Images
First we render synthetic images using [Blender](https://www.blender.org/), outputting both rendered images as well as a JSON file containing ground-truth scene information for each image.

### Installing Blender 3.6

Install Blender 3.6 LTS, then add its directory to your shell `PATH` so later commands can call `blender` directly. On Linux, download and extract the official Blender 3.6.23 tarball:

```bash
wget https://download.blender.org/release/Blender3.6/blender-3.6.23-linux-x64.tar.xz
tar -xf blender-3.6.23-linux-x64.tar.xz
```

Add the extracted Blender directory to your shell startup file:

```bash
echo "export PATH=$(pwd)/blender-3.6.23-linux-x64:\$PATH" >> ~/.bashrc
source ~/.bashrc
```

If you installed Blender somewhere else, replace `$(pwd)/blender-3.6.23-linux-x64` with that path when updating `PATH`.

Check that Blender is available:

```bash
blender --version
```

Blender ships with its own installation of Python which is used to execute scripts that interact with Blender. Blender 3.6 uses Python 3.10. In most cases `render_images.py` can now import `utils.py` directly when run from `image_generation`; if Blender cannot import it, add the `image_generation` directory to Blender's bundled Python path with a `.pth` file:

```bash
echo $PWD/image_generation >> blender-3.6.23-linux-x64/3.6/python/lib/python3.10/site-packages/clevr.pth
```

If Blender was extracted somewhere else, use that Blender directory in the `.pth` path.
For example:

```bash
echo $PWD/image_generation >> /path/to/blender-3.6.23-linux-x64/3.6/python/lib/python3.10/site-packages/clevr.pth
```

### Rendering Images

You can then render some images like this:

```bash
cd image_generation
blender --background --python render_images.py -- --num_images 10
```

To make generation reproducible and force an exact object count per image:

```bash
blender --background --python render_images.py -- --num_images 10 --random_seed 123 --num_objects 5
```

On OSX the `blender` binary is located inside the blender.app directory; for convenience you may want to add the following alias to your `~/.bash_profile` file:

```bash
alias blender='/Applications/blender/blender.app/Contents/MacOS/blender'
```

If you have an NVIDIA GPU, CUDA 12, and a driver supported by Blender 3.6, then you can use the GPU to accelerate rendering like this:

```bash
blender --background --python render_images.py -- --num_images 10 --use_gpu 1 --gpu_backend CUDA
```

On RTX cards, `--gpu_backend OPTIX` is also supported by Blender 3.6 and may be faster.

After a rendering command terminates you should have ten freshly rendered images stored in `output/images` like these:

<div align="center">
  <img src="images/img1.png" width="260px">
  <img src="images/img2.png" width="260px">
  <img src="images/img3.png" width="260px">
  <br>
  <img src="images/img4.png" width="260px">
  <img src="images/img5.png" width="260px">
  <img src="images/img6.png" width="260px">
</div>

The file `output/CLEVR_scenes.json` will contain ground-truth scene information for all newly rendered images.

You can find [more details about image rendering here](image_generation/README.md).

## Step 2: Generating Questions
Next we generate questions, functional programs, and answers for the rendered images generated in the previous step.
This step takes as input the single JSON file containing all ground-truth scene information, and outputs a JSON file 
containing questions, answers, and functional programs for the questions in a single JSON file.

You can generate questions like this:

```bash
cd question_generation
python generate_questions.py
```

The file `output/CLEVR_questions.json` will then contain questions for the generated images.

You can [find more details about question generation here](question_generation/README.md).
