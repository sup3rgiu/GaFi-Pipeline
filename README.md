# Bridging the Gap: Enhancing the Utility of Synthetic Data via Post-Processing Techniques (BMVC 2023)

[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2305.10118)

The official TensorFlow implementation of the BMVC2023 paper: [**Bridging the Gap: Enhancing the Utility of Synthetic Data via Post-Processing Techniques**](https://arxiv.org/abs/2305.10118).

![Proposed Pipeline](https://github.com/sup3rgiu/GaFi-Pipeline/assets/7725068/dafeddc7-dccd-4024-a9ad-2af87957db78)


## Results

*The CAS obtained from the classifiers trained only on generated data. The GaFi pipeline is compared with the previous state of the art, with the Synthetic Baseline and with the accuracy of the classifiers trained on real data.*

<img src="https://github.com/sup3rgiu/GaFi-Pipeline/assets/7725068/4eac8885-49ee-44c3-95fd-d0043b27b46f" width="80%">

## Installation

1. **Clone the GitHub repository:**

```shell
git clone https://github.com/sup3rgiu/GaFi-Pipeline.git
```

2. **Move inside the `docker` directory:**

```shell
cd GaFi-Pipeline/docker
```

3. **Build docker image:**

```shell
docker build --rm -t gafi_pipeline .
```

## Usage

1. **Train the classifier on real data:**

```shell
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --name GaFiPipeline -it --rm -v /path_to_GaFi-Pipeline:/exp -t gafi_pipeline python train_classifier.py --cfg_file ./configs/CIFAR10/ResNet20.yaml
```

2. **Train GAN:**

```shell
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --name GaFiPipeline -it --rm -v /path_to_GaFi-Pipeline:/exp -t gafi_pipeline python train_gain.py --cfg_file ./configs/CIFAR10/BigGAN_deep.yaml
```

3. **Run full pipeline:**\
*__N.B.__: adjust GAN name if needed. You can do it inside the .yaml file or as cmd argument*

```shell
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --name GaFiPipeline -it --rm -v /path_to_GaFi-Pipeline:/exp -t gafi_pipeline python run_pipeline.py --cfg_file ./configs/CIFAR10/Pipeline.yaml --gan_name GAN_NAME
```

4. **Iterate through steps 2. and 3. *N* times, changing the seed each time to obtain *N* different generators:**

```shell
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --name GaFiPipeline -it --rm -v /path_to_GaFi-Pipeline:/exp -t gafi_pipeline python train_gain.py --cfg_file ./configs/CIFAR10/BigGAN_deep.yaml --seed NEW_SEED

docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --name GaFiPipeline -it --rm -v /path_to_GaFi-Pipeline:/exp -t gafi_pipeline python run_pipeline.py --cfg_file ./configs/CIFAR10/Pipeline.yaml --gan_name NEW_GAN_NAME
```

5. **Run the MultiGAN script to obtain a classifier trained on a synthetic dataset sampled from the *N* different generators:**\
*__N.B.__: adjust all the GAN names inside the .yaml file or as cmd argument*

```shell
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --name GaFiPipeline -it --rm -v /path_to_GaFi-Pipeline:/exp -t gafi_pipeline python run_multigan.py --cfg_file ./configs/CIFAR10/MultiGAN.yaml
```

<br><br>
All default parameters defined in the .yaml configuration files can be overridden by specifying the corresponding command-line arguments.\
For example, if we want to use the default ./configs/CIFAR10/ResNet20.yaml but train in mixed precision, we can do the following:
```shell
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --name GaFiPipeline -it --rm -v /path_to_GaFi-Pipeline:/exp -t gafi_pipeline python train_classifier.py --cfg_file ./configs/CIFAR10/ResNet20.yaml --mixed_precision
```
Or if we want to train the classifier without using random erasing augmentation:
```shell
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --name GaFiPipeline -it --rm -v /path_to_GaFi-Pipeline:/exp -t gafi_pipeline python train_classifier.py --cfg_file ./configs/CIFAR10/ResNet20.yaml --random_erasing False
```

All possible arguments are defined in [parser.py](utils/parser.py) and can be seen by running the scripts with the `-h` flag.


## Citation
Should you find this repository useful, please consider citing:

```bibtex
@misc{lampis2023bridging,
      title={Bridging the Gap: Enhancing the Utility of Synthetic Data via Post-Processing Techniques}, 
      author={Andrea Lampis and Eugenio Lomurno and Matteo Matteucci},
      year={2023},
      eprint={2305.10118},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
