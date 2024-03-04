import torch
import matplotlib.pyplot as plt

import torch.nn.functional as F

import numpy as np
import warnings


class RandomImageGenerator:
    """ Image generator class to generate random images with specified dimensions and properties """
    def __init__(self, w:int, h:int=None, batch:int=None, sd:float=None, decorrelate:bool=True, fft:bool=True, alpha:bool=False, channels:int=None):
        """ Initialize the image generator with specified parameters.
        Args:
            w (int): Width of the image.
            h (int, optional): Height of the image. Defaults to the width if not provided.
            batch (int, optional): Batch size. Defaults to 1 if not provided.
            sd (float, optional): Standard deviation for initializing the image tensor. Only applicable for pixel-based generation.
            decorrelate (bool, optional): Whether to decorrelate color channels. Defaults to True.
            fft (bool, optional): Use FFT-based parameters for image generation if True, else pixel-based parameters. Defaults to True.
            alpha (bool, optional): Include an alpha channel in the output if True. Defaults to False.
            channels (int, optional): Number of channels in the image. Defaults to 4 if alpha is True, otherwise 3.
        """
        self.w = w
        self.h = h or w
        self.batch = batch or 1
        self.channels = channels or (4 if alpha else 3)
        self.shape = [self.batch, self.channels, self.h, self.w]

        self.alpha = alpha


    def generate_fft_image(self, sd=None, decay_power=1):
        """ Parameterize and generate an image using 2D Fourier coefficients. It allows for the generation of images with specific spectral properties.
        Original Lucid code in Tensorflow 1: (link)[https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py#L61]

        Args:
            sd (float, optional): Standard deviation of parameter initialization noise. Defaults to 0.01 if not specified.
            decay_power (float, optional): Exponent for the frequency decay. Influence the smoothness or texture of the generated image. Defaults to 1 meaning a linear decay.
        
        Returns:
            torch.Tensor: Tensor with shape from the first argument.
        """
        sd = sd or 0.01
        batch, ch, h, w = self.shape 

        # Prepare a frequency grid that can be used in Fourier transform operations. 
        # Helps in scaling the spectrum according to its frequency components.
        freqs = RandomImageGenerator.rfft2d_freqs(h, w)

        init_val_size = (batch, ch) + freqs.shape + (2,)

        init_val = torch.normal(0, sd, size=init_val_size)
        spectrum_real_imag_t = torch.nn.Parameter(init_val)
        spectrum_t = torch.view_as_complex(spectrum_real_imag_t)

        # Scale the spectrum
        scale = 1.0 / torch.maximum(freqs, torch.tensor(1.0 / max(w, h), dtype=freqs.dtype, device=freqs.device)).pow(decay_power)

        scale *= torch.sqrt(torch.tensor(w * h).float())
        scaled_spectrum_t = scale.unsqueeze(0).unsqueeze(0) * spectrum_t 

        image_t = torch.fft.irfft2(scaled_spectrum_t, s=(h, w))

        image_t = image_t / 4.0 # kind of a magic number
        return image_t    

    @staticmethod
    def rfft2d_freqs(h, w):
        """ Computes 2D spectrum frequencies for a given image size
        Prepare a frequency grid that can be used in Fourier transform operations. 
        The grid depend only on the size of the image, not on the image's actual pixel values. 

        Args:
            h (int): Height of the image.
            w (int): Width of the image.
        Returns:
            torch.Tensor: a tensor of containing the magnitude of frequencies for each pixel in the Fourier-transformed space
                          Tensor shape: [h, w // 2 + 1] or [h, w // 2 + 2] depending on whether the width is odd or even.
        """
        fy = torch.fft.fftfreq(h).view(-1, 1)

        # The output of a real FFT (RFFT) has a different size depending on whether the input width is odd or even.
        if w % 2 == 1:
            fx = torch.fft.fftfreq(w)[:w // 2 + 2]
        else:
            fx = torch.fft.fftfreq(w)[:w // 2 + 1]
        return torch.sqrt(fx**2 + fy**2)

    @staticmethod    
    def rfft2d_freqs_numpy(h, w):
        """ Computes 2D spectrum frequencies (use rfft2d_freqs instead).
        Copy of an original Lucid code in Tensorflow 1: (link)[https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py#L48]
        """

        fy = np.fft.fftfreq(h)[:, None]
        # Odd input dimension requires one additional frequency and later cut off 1 pixel
        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]
        return np.sqrt(fx * fx + fy * fy)
    
    @staticmethod
    def visualize(tensor):
        """ Visualizes an image tensor.

        Args:
            tensor (torch.Tensor): A tensor representing the generated image(s) with dimensions [batch, channels, height, width]
                                   or [channels, height, width] for a single image.
        """
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))

        if tensor.ndim == 4:    
            # if batch of images, take the first image
            image = tensor[0].detach().cpu().numpy()
        elif tensor.ndim == 3:
            image = tensor.detach().cpu().numpy()
        else:
            raise ValueError("Tensor does not have a valid image shape")

        # RGB or Grayscale depending on the number of channels
        if image.shape[0] == 3:
            plt.imshow(np.transpose(image, (1, 2, 0)))
        elif image.shape[0] == 1:
            plt.imshow(image.squeeze(), cmap='gray')
        plt.axis('off')

        plt.show()

    def generate_pixel_image(self, sd=None, init_val=None):
        """ A naive, pixel-based image parameterization.
        Defaults to a random initialization with normal distribution, but can take a supplied init_val tensor instead.

        Args:
            shape (tuple): Shape of the resulting image, [batch, channels, height, width].
            sd (float, optional): Standard deviation of parameter initialization noise. Defaults to 0.01 if not specified.
            init_val (torch.Tensor, optional): An initial value to use instead of a random initialization. Needs
                to have the same shape as the supplied shape argument.

        Returns:
            torch.Tensor: Tensor with shape from the first argument.
        """
        if sd is not None and init_val is not None:
            warnings.warn(
                "`pixel_image_pytorch` received both an initial value and a sd argument. Ignoring sd in favor of the supplied initial value."
            )

        if init_val is None:
            sd = sd or 0.01
            init_val = torch.normal(mean=0.0, std=sd, size=self.shape)
        elif not isinstance(init_val, torch.Tensor):
            raise ValueError("`init_val` must be a torch.Tensor")
        elif init_val.shape != self.shape:
            raise ValueError("`init_val` does not match the specified shape")

        return torch.nn.Parameter(init_val, requires_grad=True)


    # def generate(self):
    #     """
    #     Generate a random image tensor based on the initialized parameters.

    #     Returns:
    #         torch.Tensor: A tensor representing the generated image(s) with dimensions [batch, channels, height, width].
    #     """
    #     param_f = fft_image if self.fft else pixel_image
    #     t = param_f(self.shape, sd=self.sd)
    #     if self.channels:
    #         output = torch.sigmoid(t)
    #     else:
    #         output = to_valid_rgb(t[:, :3, :, :], decorrelate=self.decorrelate, sigmoid=True)
    #         if self.alpha:
    #             a = torch.sigmoid(t[:, 3:, :, :])
    #             output = torch.cat([output, a], 1)  # Concatenate along the channel dimension
    #     return output

#image = RandomImageGenerator(128)
##t_image = image.generate_fft_image()
#t_image = image.generate_pixel_image(sd=1)
#image.visualize(t_image)

#print(t_image.shape)
