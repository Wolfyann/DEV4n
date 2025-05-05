"""
    Sun Irradiance Calculation Module
    Version: 2025IV29
    Ref.: 
    - https://andrewmarsh.com/articles/2019/sky-distribution/
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

    # Cas extrapolation à gauche
    mask_left = idx < 0
    if extrapolate:
        slope_left = (y[1] - y[0]) / (x[1] - x[0])
        y_new[mask_left] = y[0] + slope_left * (x_new[mask_left] - x[0])
    else:
        y_new[mask_left] = np.nan

    # Cas extrapolation à droite
    mask_right = idx >= len(x) - 2
    if extrapolate:
        slope_right = (y[-1] - y[-2]) / (x[-1] - x[-2])
        y_new[mask_right] = y[-1] + slope_right * (x_new[mask_right] - x[-1])
    else:
        y_new[mask_right] = np.nan

    # Cas interpolation quadratique
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
    # Pre-compute trig functions once
    cos_alt = np.cos(alt_rad)
    
    # Compute components using broadcasting
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
    print("Attention: scipy.spatial.transform non trouvé. Utilisation de numpy pour les rotations.")
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
    # La méthode vectorielle ci-dessous est celle qui a été validée.
    # Le bloc Scipy original contenait une tentative d'euler qui semblait
    # inconsistent avec les axes définis, mais la méthode vectorielle
    # est implémentée dans les deux cas (avec ou sans Scipy pour le rouli).
    # On conserve ici la méthode vectorielle explicite pour la clarté.

    # 1. Axe Z_C (visée) dans les coordonnées Monde (W) [X=N, Y=E, Z=Up]
    # z_c_w_z = np.sin(camera_alt_rad)
    # proj_xy = np.cos(camera_alt_rad)
    # z_c_w_x = proj_xy * np.cos(camera_heading_rad) # Composante Nord
    # z_c_w_y = proj_xy * np.sin(camera_heading_rad) # Composante Est
    # Z_C_in_W = np.array([z_c_w_x, z_c_w_y, z_c_w_z])
    
    z_c_w_z = np.sin(camera_alt_rad) # Composante Z (Up) dans le repère Monde (Correct)
    proj_xy = np.cos(camera_alt_rad) # Projection sur le plan XY (Est-Nord) (Correct)

    # Calcul des composantes X (Est) et Y (Nord) de la projection
    # Si camera_heading_rad est l'angle depuis le Nord (+Y) vers l'Est (+X)
    z_c_w_x = proj_xy * np.sin(camera_heading_rad) # --> Composante X (Est)
    z_c_w_y = proj_xy * np.cos(camera_heading_rad) # --> Composante Y (Nord)

    # Le vecteur Z de la caméra dans le repère Monde [Est, Nord, Up]
    Z_C_in_W = np.array([-z_c_w_x, z_c_w_y, z_c_w_z])
    
    #Z_C_in_W = altaz_to_unit_vector(camera_alt_rad, camera_heading_rad)

    Z_C_in_W /= np.linalg.norm(Z_C_in_W) # Devrait déjà être normé si alt est entre -pi/2 et pi/2

    # 2. Axe Y_C (haut caméra) dans les coordonnées Monde (W) - sans rouli initialement
    # On veut que Y_C soit orthogonal à Z_C et le plus "vertical" possible (aligné avec Z_W=[0,0,1]).
    Z_W = np.array([0., 0., 1.])
    # Projetons le Zénith (0,0,1) sur le plan orthogonal à Z_C.
    dot_zc_zw = np.dot( Z_W, Z_C_in_W)
    if np.abs(dot_zc_zw) > 0.999: # Si visée proche du zénith/nadir
        # Utiliser une référence horizontale stable, ex: le Nord (1,0,0) projeté sur le plan normal à Z_C
        X_W = np.array([1., 0., 0.])
        Y_C_no_roll = X_W - np.dot(X_W, Z_C_in_W) * Z_C_in_W
        # Cas spécial : si Z_C = Nord, Y_C_no_roll sera nul. Visons Est (0,1,0) dans ce cas.
        if np.linalg.norm(Y_C_no_roll) < 1e-6:
             Y_W = np.array([0., 1., 0.])
             Y_C_no_roll = Y_W - np.dot(Y_W, Z_C_in_W) * Z_C_in_W
        Y_C_no_roll /= np.linalg.norm(Y_C_no_roll) # Normaliser
    else:
        Y_C_no_roll = Z_W - dot_zc_zw * Z_C_in_W
        Y_C_no_roll /= np.linalg.norm(Y_C_no_roll) # Normaliser

    # 3. Axe X_C (droite caméra) dans les coordonnées Monde (W) - sans rouli
    # X_C = Y_C x Z_C pour un système droit [X=droite, Y=haut, Z=avant]
    X_C_no_roll = np.cross(Y_C_no_roll, Z_C_in_W)
    X_C_no_roll /= np.linalg.norm(X_C_no_roll) # Devrait déjà être normé si Y et Z sont orthonormés

    # 4. Appliquer le rouli (rotation autour de Z_C_in_W)
    if np.abs(camera_roll_rad) > 1e-9:
        if use_scipy:
            # Utilisation de Scipy avec l'axe Z_C_in_W et l'angle camera_roll_rad
            roll_rot = ScipyRotation.from_rotvec(camera_roll_rad * Z_C_in_W)
            X_C_in_W = roll_rot.apply(X_C_no_roll)
            Y_C_in_W = roll_rot.apply(Y_C_no_roll)
        else:
            # Implémentation Rodrigues (rotation axe-angle)
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

    # 5. Construire la matrice de passage Monde -> Caméra
    # La matrice R_cam_to_world a pour colonnes X_C_in_W, Y_C_in_W, Z_C_in_W
    R_cam_to_world = np.stack([X_C_in_W, Y_C_in_W, Z_C_in_W], axis=-1)
    # La matrice recherchée (World -> Cam) est l'inverse (transposée pour une rotation)
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
    # Convert world coordinates to unit vector using existing function
    vec_world = altaz_to_unit_vector(alt_rad_world, az_rad_world)
    
    # Transform to camera frame
    vec_cam = R_world_to_cam @ vec_world
    
    # Early exit if point is behind camera
    if vec_cam[2] < 1e-6:
        return None, None
    
    # Calculate theta (angle from optical axis)
    # Avoid arccos instability near +/-1 by using arctan2
    theta = np.arctan2(np.sqrt(vec_cam[0]**2 + vec_cam[1]**2), vec_cam[2])
    
    # Check if within FOV
    theta_max = np.radians(fov_deg / 2.0)
    if theta > theta_max + 1e-6:
        return None, None
    
    # Calculate azimuth in camera frame
    phi_cam = np.arctan2(vec_cam[1], vec_cam[0])
    
    # Calculate radial distance in pixels
    max_radius_px = min(width, height) / 2.0
    r_px = max_radius_px * (theta / theta_max)
    
    # Convert to image coordinates
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
    # Pre-compute constants and reshape wavelengths
    wl = wavelengths[:, np.newaxis]
    valid_cos = np.clip(cos_theta_angle, -1, 1)
    
    # Calculate sun altitude in one step
    sun_alt_deg = 90 - np.degrees(np.arccos(valid_cos))
    sun_alt_deg = np.array([sun_alt_deg]) if np.isscalar(cos_theta_angle) else sun_alt_deg

    # Optimize airmass calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        # Pre-compute exponent term
        exponent_term = np.where(sun_alt_deg > -6.07995, 6.07995 + sun_alt_deg, np.nan)
        # Calculate airmass with vectorized operations
        airmass = np.where(sun_alt_deg > 85, 
                          1.0, 
                          1 / (valid_cos + 0.50572 * exponent_term ** -1.6364))
        # Apply altitude masks
        airmass = np.where(sun_alt_deg > 0, airmass, 0.0)
        airmass = np.clip(airmass, 0, 50)

    """
        Transmittance calculations:
        Bucholtz A. Rayleigh-scattering calculations for the terrestrial atmosphere.
        Appl Opt. 1995 May 20;34(15):2765-73. doi: 10.1364/AO.34.002765. PMID: 21052423.
    """
    
    airmass_exp = airmass[np.newaxis, :]
    pressure_ratio = atm_data["pressure"] / 1013.25
    
    # Pre-compute wavelength-dependent terms
    wl_terms = 1 / (wl**4 * (1 + 0.0113 * wl**2 + 0.00013 * wl**4))
    wl_mie = (wl / 550.0)**1.3
    wl_water = (wl / 720.0)**-1.5
    wl_pressure = (wl / 550.0)**-0.5

    # Vectorized transmittance calculations
    T_rayleigh = np.exp(-airmass_exp * atm_data["bRayleigh"] * pressure_ratio * wl_terms)
    T_mie = np.exp(-airmass_exp * atm_data["bMie"] / wl_mie)
    T_aerosol = np.exp(-airmass * atm_data["aerosol_optical_depth"] * (wl / 550.0)**-1.3)
    T_ozone = np.exp(-atm_data["ozone_content"] * airmass / 1000.0)
    T_water = np.exp(-airmass * (atm_data["humidity"] / 100.0) * wl_water)
    T_pressure = np.exp(-pressure_ratio * wl_pressure)

    # Combine all transmittances
    transmittance = (T_rayleigh * T_mie * T_aerosol * T_ozone * 
                    T_water * T_pressure * atm_data["ground_reflection"])

    if return_spectral:
        # return transmittance[:, 0] if transmittance.ndim > 1 else transmittance
        return np.squeeze(transmittance)
    
    # Calculate weighted irradiance
    spectrum_weighted = transmittance * camera_response[:, np.newaxis] * solar_spectrum_am15[:, np.newaxis]
    weighted_irradiance = np.sum(spectrum_weighted, axis=0)

    # Calculate twilight attenuation more efficiently
    attenuation = np.ones_like(sun_alt_deg)
    mask_ast = (sun_alt_deg >= -18) & (sun_alt_deg < -12)
    mask_naut = (sun_alt_deg >= -12) & (sun_alt_deg < -6)
    mask_civil = (sun_alt_deg >= -6) & (sun_alt_deg < 0)
    
    attenuation[mask_ast] = (sun_alt_deg[mask_ast] + 18) / 6.0
    attenuation[mask_naut] = (sun_alt_deg[mask_naut] + 12) / 6.0
    attenuation[mask_civil] = (sun_alt_deg[mask_civil] + 6) / 6.0
    attenuation[sun_alt_deg < -18] = 0.0

    # Apply final attenuation
    weighted_irradiance_attenuated = weighted_irradiance * attenuation
    weighted_irradiance_attenuated[sun_alt_deg <= 0] = 0.0

    return np.squeeze(weighted_irradiance_attenuated)
    
def simulate_irradiance_camera_v2(sun_alt_world_rad, sun_az_world_rad, wavelengths, camera_response, solar_spectrum_am15, atm_data, resolution=15):
    """
    Simule une carte d'irradiance hémisphérique basée sur la position du soleil
    et un modèle de transmittance atmosphérique simplifié.

    Le repère utilisé pour la grille et la position du soleil est le repère Monde (X=Nord, Y=Est, Z=Up).

    Args:
        sun_alt_world_rad (float): Altitude du soleil dans le repère Monde (radians, 0=horizon, pi/2=zénith).
        sun_az_world_rad (float): Azimut du soleil dans le repère Monde (radians, 0=Nord, pi/2=Est, anti-horaire).
        wavelengths (np.ndarray): Longueurs d'onde pour les calculs spectraux.
        camera_response (np.ndarray): Réponse spectrale de la caméra.
        atm_data (dict): Données atmosphériques.
        resolution (int): Résolution de la grille d'altitude et d'azimut.

    Returns:
        tuple: (irradiance_map, cos_theta_map_flat).
               irradiance_map est la carte simulée (resolution x resolution).
               cos_theta_map_flat est la carte (aplatie) des cosinus de l'angle entre chaque pixel et le soleil.
    """
    # Créer une grille couvrant l'hémisphère supérieur dans le repère Monde (Altitude, Azimut)
    # Altitude: 0 (Horizon) à pi/2 (Zénith)
    # Azimut: 0 (Nord) à 2*pi (tour complet)
    altitude_grid, azimuth_grid = np.meshgrid(np.linspace(0, np.pi / 2, resolution), np.linspace(0, 2 * np.pi, resolution))

    # Convertir la grille Alt/Az en vecteurs Cartésiens dans le repère Monde [X=Nord, Y=Est, Z=Up]
    # CORRECTION: Utiliser cos(az) pour X (Nord) et sin(az) pour Y (Est)
    """ Attention sin cos inverse dans le code initial """
    x_world = np.cos(altitude_grid) * np.sin(azimuth_grid) # X = Nord
    y_world = np.cos(altitude_grid) * np.cos(azimuth_grid) # Y = Est
    z_world = np.sin(altitude_grid) # Z = Up
    pixel_vectors_world = np.stack((x_world.flatten(), y_world.flatten(), z_world.flatten()), axis=1)
    
    # Convertir la position du soleil en vecteur Cartésien dans le repère Monde [X=Nord, Y=Est, Z=Up]
    # CORRECTION: Utiliser cos(az) pour X (Nord) et sin(az) pour Y (Est)
    # sun_vector_world = np.array([
    #     np.cos(sun_alt_world_rad) * np.cos(sun_az_world_rad), # X (Nord) component
    #     np.cos(sun_alt_world_rad) * np.sin(sun_az_world_rad), # Y (Est) component
    #     np.sin(sun_alt_world_rad) # Z (Up) component
    # ])
    sun_vector_world = altaz_to_unit_vector(sun_alt_world_rad, sun_az_world_rad)

    # Calculer le cosinus de l'angle entre chaque direction du ciel et la direction du soleil
    # (Produit scalaire des vecteurs unitaires)
    # Note: sun_vector_world might need normalization if alt/az models allow non-unit vectors,
    # but standard spherical conversion yields unit vectors. pixel_vectors_world are unit vectors.
    cos_theta_map_flat = pixel_vectors_world @ sun_vector_world
    # Ensure cos_theta is within [-1, 1] due to floating point inaccuracies
    """ Pourquoi pas entre 0 et 1 ? """
    # cos_theta_map_flat = np.clip(cos_theta_map_flat, 0.0, 1.0)
    cos_theta_map_flat = np.clip(cos_theta_map_flat, -1.0, 1.0)


    # --- Note sur le calcul de l'irradiance ---
    # La fonction calculate_transmittance est appelée avec cos_theta_map_flat.
    # Cela signifie que l'input 'cos_theta_angle' de calculate_transmittance EST LE cos(angle entre pixel et soleil).
    # calculate_transmittance interprete cet angle comme etant (lié a) l'angle au zenith du pixel,
    # ce qui lui permet (incorrectement pour le ciel) de calculer airmass PAR PIXEL.
    # Un modèle physique correct calculerait l'airmass UNE FOIS pour le soleil,
    # puis la diffusion (Rayleigh/Mie) qui DEPEND de l'angle soleil-pixel (cos_theta_map_flat).
    # Ici, nous utilisons calculate_transmittance comme une fonction empirique qui donne
    # une "luminosité" pour chaque pixel en fonction de son angle par rapport au soleil.
    # Nous ne corrigeons PAS la logique interne de calculate_transmittance ni cette simplification physique,
    # juste l'appel avec les bons vecteurs Monde.
    # ---
    irradiance_flat = calculate_transmittance(cos_theta_map_flat, wavelengths, camera_response, solar_spectrum_am15, atm_data, return_spectral=False)

    # Remettre la carte à la forme de grille
    irradiance_map = irradiance_flat.reshape((resolution, resolution))

    # Retourne la carte simulée et les cos_theta associés (utiles si on cherche le max par exemple)
    return irradiance_map, cos_theta_map_flat

def simulate_irradiance_camera_v3(sun_alt_world_rad, sun_az_world_rad, wavelengths, camera_response, solar_spectrum_am15, atm_data, resolution=15):
    """
    Simule une carte d'irradiance hémisphérique (optimisée).

    Le repère utilisé pour la grille et la position du soleil est le repère Monde (X=Nord, Y=Est, Z=Up).

    Args:
        sun_alt_world_rad (float): Altitude du soleil dans le repère Monde (radians, 0=horizon, pi/2=zénith).
        sun_az_world_rad (float): Azimut du soleil dans le repère Monde (radians, 0=Nord, pi/2=Est, anti-horaire).
        wavelengths (np.ndarray): Longueurs d'onde pour les calculs spectraux.
        camera_response (np.ndarray): Réponse spectrale de la caméra.
        solar_spectrum_am15 (np.ndarray): Spectre solaire (ex: W/m^2/nm). Assumed shape (N_wavelengths,).
        atm_data (dict): Données atmosphériques (pour calculate_transmittance).
        resolution (int): Résolution de la grille d'altitude et d'azimut (resolution x resolution).

    Returns:
        tuple: (irradiance_map, cos_theta_map_flat).
               irradiance_map est la carte simulée (resolution x resolution).
               cos_theta_map_flat est la carte (aplatie) des cosinus de l'angle entre chaque pixel et le soleil.
    """
    # Créer la grille Alt/Az
    alt_angles = np.linspace(0, np.pi / 2, resolution)
    az_angles = np.linspace(0, 2 * np.pi, resolution)
    altitude_grid, azimuth_grid = np.meshgrid(alt_angles, az_angles, indexing='ij') # Use 'ij' indexing for consistency with typical image axis order if needed, default 'xy' might swap meshgrid axes depending on interpretation. Check what's needed.

    # --- Optimization: Pre-calculate trigonometric functions ---
    cos_alt_grid = np.cos(altitude_grid)
    sin_alt_grid = np.sin(altitude_grid)
    cos_az_grid = np.cos(azimuth_grid)
    sin_az_grid = np.sin(azimuth_grid)

    # --- Correction/Clarification: Convert grid to Cartesian vectors (X=North, Y=East, Z=Up) ---
    # Ensure this matches the output convention of altaz_to_unit_vector
    x_north = cos_alt_grid * cos_az_grid
    y_east = cos_alt_grid * sin_az_grid
    z_up = sin_alt_grid

    # Flatten and stack into (N_pixels, 3) array
    # Stacking order MUST match the sun_vector_world order
    pixel_vectors_world = np.stack(
        (x_north.flatten(), y_east.flatten(), z_up.flatten()),
        axis=1
    )

    # Convertir la position du soleil en vecteur Cartésien dans le repère Monde
    # Ensure altaz_to_unit_vector returns [North, East, Up] components
    sun_vector_world = altaz_to_unit_vector(sun_alt_world_rad, sun_az_world_rad)
    # Ensure sun_vector_world is a unit vector (should be guaranteed by conversion)
    # sun_vector_world /= np.linalg.norm(sun_vector_world) # Uncomment if normalization isn't guaranteed

    # Calculer le cosinus de l'angle entre chaque direction du ciel et la direction du soleil
    # Produit scalaire de vecteurs unitaires
    cos_theta_map_flat = pixel_vectors_world @ sun_vector_world

    # Ensure cos_theta is within [-1, 1] due to potential floating point inaccuracies
    # Clipping range is correct as angle can be > pi/2.
    cos_theta_map_flat = np.clip(cos_theta_map_flat, -1.0, 1.0)

    # --- Appel à calculate_transmittance ---
    # Comme noté, cette fonction est appelée avec cos(angle entre pixel et soleil).
    # L'interprétation physique de cet appel dépend de l'implémentation de calculate_transmittance.
    # On suppose ici que c'est le comportement désiré (simplifié).
    # L'optimisation de calculate_transmittance elle-même est hors périmètre ici.
    irradiance_flat = calculate_transmittance(
        cos_theta_map_flat,
        wavelengths,
        camera_response,
        solar_spectrum_am15,
        atm_data,
        return_spectral=False # Assuming you want integrated irradiance map
    )

    # Remettre la carte à la forme de grille
    # Ensure the reshape order matches the meshgrid indexing ('ij' usually means first index is rows/altitude)
    irradiance_map = irradiance_flat.reshape((resolution, resolution))

    return irradiance_map, cos_theta_map_flat

def simulate_irradiance_camera_physical(sun_alt_world_rad, sun_az_world_rad, wavelengths, camera_response, solar_spectrum_am15, atm_data, resolution=15):
    """
    Simule une carte d'irradiance hémisphérique basée sur la position du soleil
    et un modèle de transmittance atmosphérique.

    Le repère utilisé pour la grille et la position du soleil est le repère Monde ENU :
    X=Est, Y=Nord, Z=Up.

    Args:
        sun_alt_world_rad (float): Altitude du soleil dans le repère Monde (radians, 0=horizon, pi/2=zénith).
        sun_az_world_rad (float): Azimut du soleil dans le repère Monde (radians, 0=Nord, pi/2=Est, anti-horaire).
        wavelengths (np.ndarray): Longueurs d'onde pour les calculs spectraux.
        camera_response (np.ndarray): Réponse spectrale de la caméra.
        solar_spectrum_am15 (np.ndarray): Spectre solaire (ex: W/m^2/nm).
        atm_data (dict): Données atmosphériques.
        resolution (int): Résolution de la grille d'altitude et d'azimut.

    Returns:
        tuple: (irradiance_map, cos_theta_map_flat)
    """
    # Créer la grille Alt/Az
    alt_angles = np.linspace(0, np.pi / 2, resolution)
    az_angles = np.linspace(0, 2 * np.pi, resolution)
    altitude_grid, azimuth_grid = np.meshgrid(alt_angles, az_angles, indexing='ij')

    # Convertir en vecteurs 3D dans le repère ENU (X=Est, Y=Nord, Z=Up)
    cos_alt_grid = np.cos(altitude_grid)
    sin_alt_grid = np.sin(altitude_grid)
    cos_az_grid = np.cos(azimuth_grid)
    sin_az_grid = np.sin(azimuth_grid)
    x_east = cos_alt_grid * sin_az_grid
    y_north = cos_alt_grid * cos_az_grid
    z_up = sin_alt_grid
    pixel_vectors_world = np.stack(
        (x_east.flatten(), y_north.flatten(), z_up.flatten()),
        axis=1
    )

    # Convertir la position du soleil en vecteur Cartésien ENU
    sun_vector_world = altaz_to_unit_vector(sun_alt_world_rad, sun_az_world_rad)  # doit suivre ENU

    # Calculer le cosinus de l'angle entre chaque pixel et le Soleil
    cos_theta_map_flat = pixel_vectors_world @ sun_vector_world
    cos_theta_map_flat = np.clip(cos_theta_map_flat, -1.0, 1.0)

    # Calcul de l'airmass (une seule fois pour le Soleil)
    sun_zenith_angle_rad = np.pi / 2 - sun_alt_world_rad
    sun_cos_zenith = np.cos(sun_zenith_angle_rad)
    # On passe un scalaire à calculate_transmittance pour obtenir la transmittance solaire
    transmittance_sun = calculate_transmittance(
        sun_cos_zenith, wavelengths, camera_response, solar_spectrum_am15, atm_data, return_spectral=True
    )

    # --- Calculer la diffusion Rayleigh/Mie pour chaque pixel ---
    # Rayleigh phase function: P_R = (3/4) * (1 + cos^2(theta))
    # Mie phase function: P_M = (1 - g^2) / (1 + g^2 - 2g*cos(theta))^(3/2) (Henyey-Greenstein, g~0.8)
    # Ici, on combine les deux pour une approximation simple
    g = 0.8  # asymmetry parameter for Mie
    phase_rayleigh = (3/4) * (1 + cos_theta_map_flat**2)
    phase_mie = (1 - g**2) / (1 + g**2 - 2*g*cos_theta_map_flat)**1.5
    # Poids relatifs Rayleigh/Mie (ajustez selon vos besoins)
    w_rayleigh = 0.7
    w_mie = 0.3
    phase_function = w_rayleigh * phase_rayleigh + w_mie * phase_mie  # shape: (N_pixels,)

    # --- Calculer l'irradiance pour chaque pixel ---
    # Pour chaque pixel, l'irradiance = transmittance_sun * phase_function
    # (on suppose que la transmittance_sun est déjà pondérée par le spectre solaire et la réponse caméra)
    # On normalise la phase function pour que le maximum soit 1 (optionnel)
    phase_function /= phase_function.max()

    # Multiplie la transmittance solaire par la phase function pour chaque pixel
    # transmittance_sun: (N_wavelengths,)
    # phase_function: (N_pixels,)
    # Pour obtenir l'irradiance intégrée, on fait la somme pondérée sur les longueurs d'onde
    spectrum_weighted = (transmittance_sun[:, np.newaxis] * phase_function[np.newaxis, :])
    weighted_irradiance = np.sum(spectrum_weighted, axis=0)  # shape: (N_pixels,)

    # Remettre la carte à la forme de grille
    irradiance_map = weighted_irradiance.reshape((resolution, resolution))

    return irradiance_map, cos_theta_map_flat

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
    # --- Définir la grille de directions dans le repère Caméra ---
    # Utiliser les angles polaires (theta) par rapport à l'axe Z de la caméra
    # et les angles azimutaux (phi) dans le plan XY de la caméra.
    # theta va de 0 (axe Z+) jusqu'à FOV_rad / 2
    # phi va de 0 à 2*pi
    fov_rad = np.radians(fov_deg)
    theta_angles = np.linspace(0, fov_rad / 2.0, resolution)
    phi_angles = np.linspace(0, 2 * np.pi, resolution)
    theta_grid, phi_grid = np.meshgrid(theta_angles, phi_angles, indexing='ij') # indexing='ij' pour theta x phi

    # Convertir la grille d'angles Caméra en vecteurs unitaires Cartésiens dans le repère Caméra
    # Supposons que le repère Caméra a +Z vers l'avant, +X à droite, +Y vers le bas (convention image)
    # Ou +Y vers le haut selon votre setup - Ajustez la conversion si nécessaire.
    # Si +Z avant, +X droite, +Y HAUT (convention standard 3D vision):
    x_cam = np.sin(theta_grid) * np.cos(phi_grid)
    y_cam = np.sin(theta_grid) * np.sin(phi_grid)
    z_cam = np.cos(theta_grid)
    # Si +Z avant, +X droite, +Y BAS (convention image plane avec Y descendant):
    # x_cam = np.cos(theta_grid) * np.sin(phi_grid) # Angle phi mesuré depuis +X (vers +Y)
    # y_cam = np.cos(theta_grid) * np.cos(phi_grid) # Si Y va vers le bas, sin(phi) * -1 ? Non, phi est l'angle dans le plan XY.
    # z_cam = np.sin(theta_grid) # Angle theta mesuré depuis +Z

    # Vérification de la convention pour phi_grid vs x_cam/y_cam.
    # Si phi_grid = np.linspace(0, 2*pi) et commence à 0 le long de +X et tourne vers +Y :
    # x = sin(theta) * cos(phi), y = sin(theta) * sin(phi), z = cos(theta) est la conversion standard.
    # Si Y va vers le bas, le système reste direct si Z va vers l'avant et X vers la droite.
    # Le plan XY est juste orienté différemment. La conversion ci-dessus est correcte pour ce système.

    # Flatten et empiler dans un tableau (N_pixels, 3)
    pixel_vectors_cam = np.stack(
        (x_cam.flatten(), y_cam.flatten(), z_cam.flatten()),
        axis=1
    ) # (N_pixels, 3), N_pixels = resolution * resolution

    # --- Transformer la position du soleil du repère Monde ENU vers le repère Caméra ---
    # Convertir la position du soleil en vecteur Cartésien dans le repère Monde ENU
    # altaz_to_unit_vector DOIT retourner le vecteur dans le repère Monde ENU (X=Est, Y=Nord, Z=Up)
    sun_vector_world = altaz_to_unit_vector(sun_alt_world_rad, sun_az_world_rad)

    # Appliquer la matrice de rotation Monde->Caméra
    sun_vector_cam = R_world_to_cam @ sun_vector_world
    # S'assurer que c'est un vecteur unitaire (devrait l'être après rotation d'un unitaire)
    # sun_vector_cam /= np.linalg.norm(sun_vector_cam) # Décommenter si besoin

    # --- Calculer le cosinus de l'angle entre chaque direction du ciel (pixel)
    # et la direction du soleil, dans le repère Caméra ---
    # Produit scalaire de vecteurs unitaires, maintenant tous deux dans le repère Caméra
    cos_theta_map_flat = pixel_vectors_cam @ sun_vector_cam
    cos_theta_map_flat = np.clip(cos_theta_map_flat, -1.0, 1.0) # Clamper pour robustesse

    # --- Calcul de l'airmass et de la transmittance solaire directe ---
    # L'airmass dépend de l'angle zénithal du soleil dans le repère Monde (par rapport à la verticale locale Z Up)
    sun_zenith_angle_world_rad = np.pi / 2 - sun_alt_world_rad
    sun_cos_zenith_world = np.cos(sun_zenith_angle_world_rad)

    # On passe le cos(angle zénithal du soleil Monde) à calculate_transmittance
    # pour obtenir la transmittance solaire à travers l'atmosphère (le chemin le plus court vers le soleil).
    transmittance_sun = calculate_transmittance(
        sun_cos_zenith_world, wavelengths, camera_response, solar_spectrum_am15,
        atm_data, return_spectral=True # On veut la transmittance spectrale du soleil
    )

    # --- Calculer la diffusion Rayleigh/Mie pour chaque pixel (en fonction de l'angle de diffusion) ---
    # L'angle de diffusion est l'angle entre le pixel (la direction observée) et le soleil.
    # C'est exactement ce que cos_theta_map_flat représente.
    g = 0.8  # paramètre d'asymétrie pour Mie
    phase_rayleigh = (3/4) * (1 + cos_theta_map_flat**2)
    # Éviter la division par zéro si 1 + g^2 - 2*g*cos_theta_map_flat est très proche de zéro
    # (arrive pour cos_theta_map_flat très proche de 1/g quand g~1)
    denominator_mie = (1 + g**2 - 2*g*cos_theta_map_flat)
    phase_mie = (1 - g**2) / np.where(denominator_mie > 1e-9, denominator_mie**1.5, 1e-9**1.5) # Petite valeur pour éviter l'infini

    # Poids relatifs Rayleigh/Mie (ajustez selon votre modèle atmosphérique)
    w_rayleigh = 0.7
    w_mie = 0.3
    phase_function = w_rayleigh * phase_rayleigh + w_mie * phase_mie  # shape: (N_pixels,)

    # --- Calculer l'irradiance pour chaque pixel ---
    # L'irradiance du ciel pour un pixel est (approximativement) la lumière solaire transmise
    # multipliée par la fonction de phase pour l'angle de diffusion de ce pixel.
    # transmittance_sun: (N_wavelengths,)
    # phase_function: (N_pixels,)
    # Pour obtenir l'irradiance intégrée sur les longueurs d'onde, on pondère la transmittance
    # spectrale du soleil par la phase function (qui est spectrale si les poids w_rayleigh/w_mie le sont,
    # mais ici c'est une approximation simple indépendante des longueurs d'onde pour la phase)
    # et on somme sur les longueurs d'onde.

    # On multiplie la transmittance solaire (spectrale) par la phase function (scalaire pour chaque pixel)
    # Chaque colonne de spectrum_weighted correspond au spectre d'irradiance pour un pixel.
    spectrum_weighted = transmittance_sun[:, np.newaxis] * phase_function[np.newaxis, :] # Shape (N_wavelengths, N_pixels)

    # Intégrer sur les longueurs d'onde pour obtenir l'irradiance totale par pixel
    # Cette somme représente l'irradiance pondérée par la réponse caméra et le spectre solaire AM1.5
    weighted_irradiance = np.sum(spectrum_weighted, axis=0)  # shape: (N_pixels,)

    # Remettre la carte d'irradiance à la forme de grille (correspondant à theta x phi caméra)
    # L'ordre doit correspondre à l'indexation de meshgrid ('ij')
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
    # Le rayon max dans l'image fisheye
    max_radius = min(center_x, center_y)

    alt_steps, az_steps = output_shape

    # Angle maximal du fisheye mappé au rayon max (en degrés)
    fov_deg_half = fov_deg / 2.0
    if fov_deg_half <= 1e-6:
         print("Erreur: fov_deg doit être positif.")
         return np.zeros((alt_steps, az_steps, image.shape[2] if image.ndim == 3 else 1), dtype=image.dtype)

    # --- Vectorisation des calculs ---

    # Crée des grilles de coordonnées pour l'image panoramique de sortie
    # j: indice horizontal (azimut), i: indice vertical (altitude)
    j_grid, i_grid = np.meshgrid(np.arange(az_steps), np.arange(alt_steps))

    # 1. Calculer l'altitude et l'azimut standards du panorama pour chaque pixel
    # Altitude: de +90 (i=0) à -90 (i=alt_steps-1)
    alt_pano_deg_grid = 90.0 * i_grid / (alt_steps - 1.0)
    # Azimut: de 0 (j=0) à <360 (j=az_steps-1)
    az_pano_deg_grid = 360.0 * j_grid / (az_steps - 1.0)

    # 2. Appliquer le mapping inverse de l'altitude pour obtenir l'angle theta (en degrés)
    # Inverse de : altitude_pano = 90 - theta + camera_alt
    # Donc : theta = 90 - altitude_pano + camera_alt
    theta_deg_grid =  90 - alt_pano_deg_grid + (90 - camera_alt_deg)

    # 3. Gérer les points du panorama qui tombent en dehors du champ de vision du fisheye
    # Cela correspond aux points où theta < 0 ou theta > fov_deg_half
    mask = (theta_deg_grid >= 1e-6) & (theta_deg_grid <= fov_deg_half)

    # 4. Calculer le rayon correspondant dans l'image fisheye (projection équidistante)
    # r = (theta / theta_max) * max_radius
    radius_px_grid = np.zeros_like(theta_deg_grid)
    radius_px_grid[mask] = (theta_deg_grid[mask] / fov_deg_half) * max_radius

    # 5. L'angle phi dans le plan du fisheye est directement l'azimut panoramique
    # (selon la logique du snippet solaire, azimut 0=360 aligné avec l'axe +X du fisheye, croissant anti-horaire)
    phi_fish_rad_grid = np.radians(az_pano_deg_grid)

    # 6. Convertir les coordonnées polaires (radius_px, phi_fish_rad) en cartésiennes (dx, dy)
    # Note: np.cos/sin attend des radians. phi_fish_rad_grid va de 0 à 2*pi.
    # dx = r * cos(phi), dy = r * sin(phi) pour angle CCW depuis +X
    dx_grid = radius_px_grid * np.cos(phi_fish_rad_grid)
    dy_grid = radius_px_grid * np.sin(phi_fish_rad_grid)

    # 7. Calculer les coordonnées finales (x, y) dans l'image fisheye centrées
    x_fisheye_grid = center_x + dx_grid
    y_fisheye_grid = center_y + dy_grid

    # --- Remappage avec OpenCV ---

    # cv2.remap attend les coordonnées en float32
    map_x = x_fisheye_grid.astype(np.float32)
    map_y = y_fisheye_grid.astype(np.float32)

    # Pour les points masqués (hors champ du fisheye), définir des coordonnées invalides
    # (-1, -1 est une convention courante pour cv2.remap avec BORDER_CONSTANT)
    map_x[~mask] = -1.0
    map_y[~mask] = -1.0

        # *** CORRECTION DE L'INVERSION VERTICALE ***
    # L'inversion vient du mapping vertical. Pour corriger l'orientation de l'image de sortie,
    # nous inversons simplement l'ordre des lignes dans les maps de coordonnées.
    # Cela signifie que le pixel calculé pour la ligne 'i' (qui correspond à l'altitude 90 - 180*i/...)
    # sera placé à la ligne 'alt_steps - 1 - i' dans l'image finale.
    # map_x_flipped = map_x[::-1, :] # Inverse les lignes de map_x
    # map_y_flipped = map_y[::-1, :] # Inverse les lignes de map_y

    # Effectue le remappage. cv2.remap gère l'interpolation et les bords.
    # INTER_LINEAR NEAREST CUBIC pour l'interpolation bilinéaire
    # BORDER_CONSTANT avec 0,0,0 pour les pixels hors du cercle fisheye ou masqués (noir)
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
