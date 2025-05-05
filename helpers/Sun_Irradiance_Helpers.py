"""
    Sun Irradiance Calculation Module
    Version: 2025IV29
"""

import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from datetime import datetime
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
from astropy.time import TimeDelta
import astropy.units as u
import matplotlib.gridspec as gridspec
import sys

def quadratic_interp(x, y, x_new, extrapolate=True):
    x = np.asarray(x)
    y = np.asarray(y)
    x_new = np.asarray(x_new)
    y_new = np.empty_like(x_new, dtype=float)

    idx = np.searchsorted(x, x_new) - 1

    mask_left = idx < 0
    if extrapolate:
        slope_left = (y[1] - y[0]) / (x[1] - x[0])
        y_new[mask_left] = y[0] + slope_left * (x_new[mask_left] - x[0])
    else:
        y_new[mask_left] = np.nan

    mask_right = idx >= len(x) - 2
    if extrapolate:
        slope_right = (y[-1] - y[-2]) / (x[-1] - x[-2])
        y_new[mask_right] = y[-1] + slope_right * (x_new[mask_right] - x[-1])
    else:
        y_new[mask_right] = np.nan

    mask_mid = ~(mask_left | mask_right)
    i = idx[mask_mid]
    x1, x2, x3 = x[i], x[i+1], x[i+2]
    y1, y2, y3 = y[i], y[i+1], y[i+2]
    val = x_new[mask_mid]
    a = y1
    b = (y2 - y1) / (x2 - x1)
    c = ((y3 - y2) / (x3 - x2) - (y2 - y1) / (x2 - x1)) / (x3 - x1)
    y_new[mask_mid] = a + b * (val - x1) + c * (val - x1) * (val - x2)

    return y_new

def altaz_to_unit_vector(alt_rad, az_rad):
    """
    Vectorized conversion from Alt-Az coordinates to unit vector in World frame.
    
    Args:
        alt_rad (float or np.ndarray): Altitude in radians (0=horizon, pi/2=zenith)
        az_rad (float or np.ndarray): Azimuth in radians (0=North, pi/2=East)

    Returns:
        np.ndarray: Unit vector [X=East, Y=North, Z=Up]
    """

    cos_alt = np.cos(alt_rad)

    return np.array([
        cos_alt * np.sin(az_rad),    # East
        cos_alt * np.cos(az_rad),    # North
        np.sin(alt_rad)              # Up
    ])

try:
    from scipy.spatial.transform import Rotation as ScipyRotation
    use_scipy = True
except ImportError:
    use_scipy = False
    print("Warning: scipy.spatial.transform not found")
def build_world_to_camera_matrix(camera_heading_rad, camera_alt_rad, camera_roll_rad):
    """
    Calcule la matrice de rotation pour passer des coordonnées Monde (X=N, Y=E, Z=Up)
    aux coordonnées Caméra (X'=droite, Y'=haut, Z'=avant).

    Args:
        camera_heading_rad (float): Azimut de visée (radians, 0=N, pi/2=E, anti-horaire).
        camera_alt_rad (float): Altitude de visée (radians, 0=Horizon, pi/2=Zénith).
        camera_roll_rad (float): Rouli autour de l'axe de visée Z' (radians).

    Returns:
        np.ndarray: Matrices de rotation 3x3 (R_world_to_cam et da transposée).
    """
    
    z_c_w_z = np.sin(camera_alt_rad)
    proj_xy = np.cos(camera_alt_rad)

    z_c_w_x = proj_xy * np.sin(camera_heading_rad)
    z_c_w_y = proj_xy * np.cos(camera_heading_rad)

    Z_C_in_W = np.array([-z_c_w_x, z_c_w_y, z_c_w_z])
    
    Z_C_in_W /= np.linalg.norm(Z_C_in_W) 
    
    Z_W = np.array([0., 0., 1.])
    
    dot_zc_zw = np.dot( Z_W, Z_C_in_W)
    if np.abs(dot_zc_zw) > 0.999:
        
        X_W = np.array([1., 0., 0.])
        Y_C_no_roll = X_W - np.dot(X_W, Z_C_in_W) * Z_C_in_W
       
        if np.linalg.norm(Y_C_no_roll) < 1e-6:
             Y_W = np.array([0., 1., 0.])
             Y_C_no_roll = Y_W - np.dot(Y_W, Z_C_in_W) * Z_C_in_W
        Y_C_no_roll /= np.linalg.norm(Y_C_no_roll)
    else:
        Y_C_no_roll = Z_W - dot_zc_zw * Z_C_in_W
        Y_C_no_roll /= np.linalg.norm(Y_C_no_roll)

    X_C_no_roll = np.cross(Y_C_no_roll, Z_C_in_W)
    X_C_no_roll /= np.linalg.norm(X_C_no_roll) 
    
    if np.abs(camera_roll_rad) > 1e-9:
        if use_scipy:
            roll_rot = ScipyRotation.from_rotvec(camera_roll_rad * Z_C_in_W)
            X_C_in_W = roll_rot.apply(X_C_no_roll)
            Y_C_in_W = roll_rot.apply(Y_C_no_roll)
        else:
            axis = Z_C_in_W
            angle = camera_roll_rad
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R_roll_mat = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            X_C_in_W = R_roll_mat @ X_C_no_roll
            Y_C_in_W = R_roll_mat @ Y_C_no_roll
    else:
        X_C_in_W = X_C_no_roll
        Y_C_in_W = Y_C_no_roll

    R_cam_to_world = np.stack([X_C_in_W, Y_C_in_W, Z_C_in_W], axis=-1)
    R_world_to_cam = R_cam_to_world.T

    return R_cam_to_world, R_world_to_cam

def fisheye_projection_equidistant(alt_rad_world, az_rad_world, width, height, fov_deg, R_world_to_cam):
    """
    Projects world Alt/Az coordinates onto fisheye image plane using equidistant model.
    Vectorized implementation for better performance.
    
    Args:
        alt_rad_world: Altitude in world frame (radians, 0=horizon, pi/2=zenith)
        az_rad_world: Azimuth in world frame (radians, 0=North, pi/2=East)
        width, height: Image dimensions in pixels
        fov_deg: Total field of view in degrees
        R_world_to_cam: 3x3 rotation matrix from world to camera frame
    
    Returns:
        tuple: (x_img, y_img) pixel coordinates or (None, None) if out of view
    """
    vec_world = altaz_to_unit_vector(alt_rad_world, az_rad_world)

    vec_cam = R_world_to_cam @ vec_world

    if vec_cam[2] < 1e-6:
        return None, None

    theta = np.arctan2(np.sqrt(vec_cam[0]**2 + vec_cam[1]**2), vec_cam[2])

    theta_max = np.radians(fov_deg / 2.0)
    if theta > theta_max + 1e-6:
        return None, None

    phi_cam = np.arctan2(vec_cam[1], vec_cam[0])

    max_radius_px = min(width, height) / 2.0
    r_px = max_radius_px * (theta / theta_max)

    x_img = (width / 2.0) + r_px * np.cos(phi_cam)
    y_img = (height / 2.0) - r_px * np.sin(phi_cam)
    
    return x_img, y_img

def calculate_transmittance(cos_theta_angle, wavelengths, camera_response, solar_spectrum_am15, atm_data, return_spectral=False):
    """
    Calculate atmospheric transmittance with optimized vectorized operations.
    
    Args:
        cos_theta_angle: Cosine of angle between pixel and sun
        wavelengths: Array of wavelengths for spectral calculations
        camera_response: Camera spectral response
        solar_spectrum_am15: Solar spectrum at AM1.5
        atm_data: Dictionary of atmospheric parameters
        return_spectral: If True, return spectral transmittance
    
    Returns:
        Transmittance values (spectral or integrated)
    """

    wl = wavelengths[:, np.newaxis]
    valid_cos = np.clip(cos_theta_angle, -1, 1)

    sun_alt_deg = 90 - np.degrees(np.arccos(valid_cos))
    sun_alt_deg = np.array([sun_alt_deg]) if np.isscalar(cos_theta_angle) else sun_alt_deg

    with np.errstate(divide='ignore', invalid='ignore'):
        exponent_term = np.where(sun_alt_deg > -6.07995, 6.07995 + sun_alt_deg, np.nan)
        airmass = np.where(sun_alt_deg > 85, 
                          1.0, 
                          1 / (valid_cos + 0.50572 * exponent_term ** -1.6364))
        airmass = np.where(sun_alt_deg > 0, airmass, 0.0)
        airmass = np.clip(airmass, 0, 50)
    
    airmass_exp = airmass[np.newaxis, :]
    pressure_ratio = atm_data["pressure"] / 1013.25
    
    wl_terms = 1 / (wl**4 * (1 + 0.0113 * wl**2 + 0.00013 * wl**4))
    wl_mie = (wl / 550.0)**1.3
    wl_water = (wl / 720.0)**-1.5
    wl_pressure = (wl / 550.0)**-0.5

    T_rayleigh = np.exp(-airmass_exp * atm_data["bRayleigh"] * pressure_ratio * wl_terms)
    T_mie = np.exp(-airmass_exp * atm_data["bMie"] / wl_mie)
    T_aerosol = np.exp(-airmass * atm_data["aerosol_optical_depth"] * (wl / 550.0)**-1.3)
    T_ozone = np.exp(-atm_data["ozone_content"] * airmass / 1000.0)
    T_water = np.exp(-airmass * (atm_data["humidity"] / 100.0) * wl_water)
    T_pressure = np.exp(-pressure_ratio * wl_pressure)

    transmittance = (T_rayleigh * T_mie * T_aerosol * T_ozone * 
                    T_water * T_pressure * atm_data["ground_reflection"])

    if return_spectral:
        return np.squeeze(transmittance)

    spectrum_weighted = transmittance * camera_response[:, np.newaxis] * solar_spectrum_am15[:, np.newaxis]
    weighted_irradiance = np.sum(spectrum_weighted, axis=0)

    attenuation = np.ones_like(sun_alt_deg)
    mask_ast = (sun_alt_deg >= -18) & (sun_alt_deg < -12)
    mask_naut = (sun_alt_deg >= -12) & (sun_alt_deg < -6)
    mask_civil = (sun_alt_deg >= -6) & (sun_alt_deg < 0)
    
    attenuation[mask_ast] = (sun_alt_deg[mask_ast] + 18) / 6.0
    attenuation[mask_naut] = (sun_alt_deg[mask_naut] + 12) / 6.0
    attenuation[mask_civil] = (sun_alt_deg[mask_civil] + 6) / 6.0
    attenuation[sun_alt_deg < -18] = 0.0
    
    weighted_irradiance_attenuated = weighted_irradiance * attenuation
    weighted_irradiance_attenuated[sun_alt_deg <= 0] = 0.0

    return np.squeeze(weighted_irradiance_attenuated)

def simulate_irradiance_camera_physical_cam(
    sun_alt_world_rad, sun_az_world_rad, R_world_to_cam, fov_deg,
    wavelengths, camera_response, solar_spectrum_am15, atm_data,
    resolution=15
):
    """
    Simule une carte d'irradiance dans le repère de la caméra.

    La grille de simulation et les calculs sont effectués dans le repère Caméra.
    La position du soleil est transformée du repère Monde ENU vers le repère Caméra.

    Args:
        sun_alt_world_rad (float): Altitude du soleil dans le repère Monde ENU (radians).
        sun_az_world_rad (float): Azimut du soleil dans le repère Monde ENU (radians).
        R_world_to_cam (np.ndarray): Matrice de rotation 3x3 du repère Monde ENU vers le repère Caméra.
        fov_deg (float): Champ de vision total de la caméra en degrés.
        wavelengths (np.ndarray): Longueurs d'onde pour les calculs spectraux.
        camera_response (np.ndarray): Réponse spectrale de la caméra.
        solar_spectrum_am15 (np.ndarray): Spectre solaire (ex: W/m^2/nm).
        atm_data (dict): Données atmosphériques.
        resolution (int): Résolution de la grille angulaire dans le repère caméra (resolution x resolution).

    Returns:
        tuple: (irradiance_map, cos_theta_map_flat).
               irradiance_map est la carte simulée (resolution x resolution), où les axes
               correspondent aux angles polaires et azimutaux dans le repère Caméra.
               cos_theta_map_flat est la carte (aplatie) des cosinus de l'angle entre chaque
               direction du pixel et la direction du soleil (dans le repère Caméra).
    """

    fov_rad = np.radians(fov_deg)
    theta_angles = np.linspace(0, fov_rad / 2.0, resolution)
    phi_angles = np.linspace(0, 2 * np.pi, resolution)
    theta_grid, phi_grid = np.meshgrid(theta_angles, phi_angles, indexing='ij')

    x_cam = np.sin(theta_grid) * np.cos(phi_grid)
    y_cam = np.sin(theta_grid) * np.sin(phi_grid)
    z_cam = np.cos(theta_grid)

    pixel_vectors_cam = np.stack(
        (x_cam.flatten(), y_cam.flatten(), z_cam.flatten()),
        axis=1
    )

    sun_vector_world = altaz_to_unit_vector(sun_alt_world_rad, sun_az_world_rad)

    sun_vector_cam = R_world_to_cam @ sun_vector_world
    cos_theta_map_flat = pixel_vectors_cam @ sun_vector_cam
    cos_theta_map_flat = np.clip(cos_theta_map_flat, -1.0, 1.0)
    sun_zenith_angle_world_rad = np.pi / 2 - sun_alt_world_rad
    sun_cos_zenith_world = np.cos(sun_zenith_angle_world_rad)

    transmittance_sun = calculate_transmittance(
        sun_cos_zenith_world, wavelengths, camera_response, solar_spectrum_am15,
        atm_data, return_spectral=True 
    )

    g = 0.8 
    phase_rayleigh = (3/4) * (1 + cos_theta_map_flat**2)

    denominator_mie = (1 + g**2 - 2*g*cos_theta_map_flat)
    phase_mie = (1 - g**2) / np.where(denominator_mie > 1e-9, denominator_mie**1.5, 1e-9**1.5)
    w_rayleigh = 0.7
    w_mie = 0.3
    phase_function = w_rayleigh * phase_rayleigh + w_mie * phase_mie

    spectrum_weighted = transmittance_sun[:, np.newaxis] * phase_function[np.newaxis, :]5
    weighted_irradiance = np.sum(spectrum_weighted, axis=0) 
    irradiance_map = weighted_irradiance.reshape((resolution, resolution))

    return irradiance_map, cos_theta_map_flat

def fisheye_to_panorama(image, fov_deg=180, output_shape=(90, 360), camera_alt_deg=90):
    """
    Convertit une image fisheye en image panoramique (equirectangulaire).

    Utilise la projection fisheye équidistante et un mapping altitude/azimut
    compatible avec le code de calcul de position du soleil fourni, incluant
    un décalage vertical (altitude) de la caméra.

    Optimisé par vectorisation et utilisation de cv2.remap.

    Args:
        image (np.ndarray): L'image fisheye d'entrée (BGR ou niveaux de gris).
        fov_deg (float): Le champ de vision total du fisheye en degrés.
                         (fov_deg / 2.0 est l'angle mappé au rayon max dans le modèle équidistant).
        output_shape (tuple): La forme (hauteur, largeur) de l'image panoramique de sortie.
                              Assume mapping: i -> Altitude [90, -90], j -> Azimut [0, 360).
        camera_alt_deg (float): L'altitude (en degrés dans le repère panoramique)
                                de l'axe optique de la caméra fisheye. Permet un décalage vertical.

    Returns:
        np.ndarray: L'image panoramique de sortie.
    """
    if image is None:
        print("Erreur: L'image d'entrée est vide ou invalide.")
        return None

    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2

    max_radius = min(center_x, center_y)

    alt_steps, az_steps = output_shape

    fov_deg_half = fov_deg / 2.0
    if fov_deg_half <= 1e-6:
         print("Erreur: fov_deg doit être positif.")
         return np.zeros((alt_steps, az_steps, image.shape[2] if image.ndim == 3 else 1), dtype=image.dtype)

    j_grid, i_grid = np.meshgrid(np.arange(az_steps), np.arange(alt_steps))

    alt_pano_deg_grid = 90.0 * i_grid / (alt_steps - 1.0)
    az_pano_deg_grid = 360.0 * j_grid / (az_steps - 1.0)

    theta_deg_grid =  90 - alt_pano_deg_grid + (90 - camera_alt_deg)

    mask = (theta_deg_grid >= 1e-6) & (theta_deg_grid <= fov_deg_half)

    radius_px_grid = np.zeros_like(theta_deg_grid)
    radius_px_grid[mask] = (theta_deg_grid[mask] / fov_deg_half) * max_radius

    phi_fish_rad_grid = np.radians(az_pano_deg_grid)

    dx_grid = radius_px_grid * np.cos(phi_fish_rad_grid)
    dy_grid = radius_px_grid * np.sin(phi_fish_rad_grid)

    x_fisheye_grid = center_x + dx_grid
    y_fisheye_grid = center_y + dy_grid

    map_x = x_fisheye_grid.astype(np.float32)
    map_y = y_fisheye_grid.astype(np.float32)

    map_x[~mask] = -1.0
    map_y[~mask] = -1.0

    panorama = cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return panorama

def get_spectral_irradiance_at_max_point(irradiance_map, cos_theta_angle_flat, wavelengths, camera_response, atm_data, solar_spectrum_am15):
    max_idx = np.argmax(irradiance_map)
    max_cos = cos_theta_angle_flat[max_idx]
    transmittance = calculate_transmittance(max_cos, wavelengths, camera_response, solar_spectrum_am15, atm_data, return_spectral=True)
    spectral_irradiance = transmittance * solar_spectrum_am15 * camera_response
    max_spectral = np.max(spectral_irradiance)
    return wavelengths, spectral_irradiance / max_spectral if max_spectral > 1e-6 else spectral_irradiance
