# Denoising Audio using HiFi-Coulomb-GANs
## Anirudh Satheesh, Karthick Muthu-Manivannan

In our paper, we propose HiFi-Coulomb-GAN, a Generative Adversarial Network model that combines the approaches of HiFi-GAN by Su et al. and CoulombGAN by Unterthiner et al. for audio denoising and speech enhancement. We provide an open-source implementation of our code in this repository along with a pretrained model.

**Abstract**: Recorded speech signals often contain noise that affects the quality of the signal and reduces intelligibility. Several studies have used Generative Adversarial Networks (GANs) to remove noise artifacts and improve speech intelligibility. However, GANs can suffer from gradient vanishing or gradient explosion that can reduce their effectiveness in denoising. To mitigate gradient vanishing, we applied the CoulombGAN architecture to speech denoising using a model structure similar to Hifi-GAN, the current state of the art speech denoiser. We call this new model Hifi-CoGAN. We used a WaveNet generator to denoise signals, a PostNet for general cleanup, and a Multi-Resolution Discriminator to evaluate the signal quality relative to the clean signal. Our results show that Hifi-CoGAN was able to outperform Hifi-GAN in many of the narrowband signals (signals with a limited range of frequencies) in terms of the Short-Term Objective Intelligibility (STOI) and Perceptual Evaluation of Speech Quality (PESQ) metrics. However, the model did not perform as well as Hifi-GAN with wideband noise signals (signals with a wider range of frequencies) such as white noise, so future work must be done to improve the model for these noise signals.

## Requirements
1. Python 3.7+
2. Install the required python packages.
3. Download the MS-SNSD dataset and move it to the wavs folder.

## Training
```
python train.py -c config.yaml
```
## Inference
1. Use test files from the MS-SNSD dataset or use your own.
2. Run the following command:
```
python inference.py --checkpoint_file [generator checkpoint file]
```
