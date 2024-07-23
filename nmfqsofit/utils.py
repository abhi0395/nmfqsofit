# nmfqsofit/utils.py

import numpy as np
import matplotlib.pyplot as plt

def plot_flux_and_continuum(flux_matrix, continuum_matrix):
    index = np.random.randint(flux_matrix.shape[0])
    flux = flux_matrix[index]
    continuum = continuum_matrix[index]

    plt.figure(figsize=(10, 6))
    plt.plot(flux, label='Flux')
    plt.plot(continuum, label='Continuum', linestyle='--')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title(f'Flux and Continuum for Index {index}')
    plt.legend()
    plt.show()
