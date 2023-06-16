## Replica Data Generation

### Download Replica Dataset
Download 3D models and info files from [Replica](https://github.com/facebookresearch/Replica-Dataset)

### 3D Object Mesh Extraction
Change the input path in `./data_generation/extract_inst_obj.py` and run
```angular2html
python ./data_generation/extract_inst_obj.py
```

### Camera Trajectory Generation
Please refer to [Semantic-NeRF](https://github.com/Harry-Zhi/semantic_nerf/issues/25#issuecomment-1340595427) for more details. The random trajectory generation only works for single room scene. For multiple rooms scene, collision checking is needed. Welcome contributions.

### Rendering 2D Images
Given camera trajectory t_wc (change pose_file in configs), we use [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) to render RGB, Depth, Semantic and Instance images.

#### Install Habitat-Sim 0.2.1
We recommend to use conda to install habitat-sim 0.2.1.
```angular2html
conda create -n habitat python=3.8.12 cmake=3.14.0
conda activate habitat
conda install habitat-sim=0.2.1 withbullet -c conda-forge -c aihabitat 
conda install numba=0.54.1
```

#### Run rendering with configs
```angular2html
python ./data_generation/habitat_renderer.py --config ./data_generation/replica_render_config_vMAP.yaml 
```
Note that to get HDR img, use mesh.ply not semantic_mesh.ply. Change path in configs. And copy rgb folder to replace previous high exposure rgb.
```angular2html
python ./data_generation/habitat_renderer.py --config ./data_generation/replica_render_config_vMAP.yaml 
```