"""
    Test Simulation Irradiance Solaire
"""

import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time, TimeDelta
import astropy.units as u
import imageio.v2 as imageio
import os
import sys

from Sun_Irradiance_Helpers import (
    quadratic_interp,
    altaz_to_unit_vector,
    build_world_to_camera_matrix,
    fisheye_projection_equidistant,
    # simulate_irradiance_camera_v2,
    # simulate_irradiance_camera_v3,
    # simulate_irradiance_camera_physical,
    simulate_irradiance_camera_physical_cam,
    fisheye_to_panorama,
    get_spectral_irradiance_at_max_point
)

def interpolate_sky_spectrum_to_camera(parameters, resolution_spectral):
    """
    Interpole le spectre du ciel depuis les données chargées dans 'parameters'
    sur la plage de longueurs d'onde et avec le nombre d'échantillons
    définis par les données de la caméra.

    Args:
        parameters (dict): Le dictionnaire contenant les données chargées depuis le fichier JSON.
        resolution_spectral (float): La résolution spectrale de la caméra.

    Returns:
        tuple: Un tuple contenant les longueurs d'onde interpolées (np.ndarray)
               et l'irradiance interpolée et normalisée (np.ndarray),
               ou (None, None) en cas d'erreur.
    """
    sky_data = parameters.get("sky_model")
    if not sky_data or not sky_data.get("spectrum"):
        print("Erreur: La clé 'sky_model' ou 'sky_model.spectrum' est absente des paramètres.")
        return None, None

    spectrum_data = sky_data["spectrum"]
    wavelengths_json = np.array([int(k) for k in spectrum_data.keys()])
    irradiance_json = np.array(list(spectrum_data.values()))

    # Assurez-vous que les longueurs d'onde du ciel sont triées
    sort_indices_sky = np.argsort(wavelengths_json)
    wavelengths_json = wavelengths_json[sort_indices_sky]
    irradiance_json = irradiance_json[sort_indices_sky]

    camera_data = parameters.get("camera")
    if not camera_data or not camera_data.get("spectral_response"):
        print("Erreur: La clé 'camera' ou 'camera.spectral_response' est absente des paramètres.")
        return None, None

    spectral_response_dict = {int(k): v for k, v in camera_data["spectral_response"].items()}
    wavelengths_camera = np.array(list(spectral_response_dict.keys()), dtype=int)
    min_wavelength_camera = np.min(wavelengths_camera)
    max_wavelength_camera = np.max(wavelengths_camera)

    nm_samples = int((max_wavelength_camera - min_wavelength_camera) / resolution_spectral) + 1
    if nm_samples < 2:
        nm_samples = 2

    # Définir la nouvelle plage de longueurs d'onde pour l'interpolation (basée sur la caméra)
    wavelengths_spaced = np.linspace(min_wavelength_camera, max_wavelength_camera, nm_samples)
    irradiance_interpolated = quadratic_interp(wavelengths_json, irradiance_json, wavelengths_spaced, extrapolate=True)

    # Normalisation de l'irradiance interpolée entre 0 et 1
    irradiance_interpolated_normalized = (irradiance_interpolated - np.min(irradiance_interpolated)) / (np.max(irradiance_interpolated) - np.min(irradiance_interpolated))
    irradiance_interpolated_normalized /= np.max(irradiance_interpolated_normalized)

    return wavelengths_spaced, irradiance_interpolated_normalized

def draw_fisheye_grid(ax, width, height, fov_deg=180, step_deg=15):
    center_x, center_y = width // 2, height // 2
    max_radius = min(center_x, center_y)
    theta_max_rad = np.radians(fov_deg / 2.0)

    for alt_deg in range(step_deg, 90, step_deg):
        alt_rad_world = np.radians(alt_deg)
        r = max_radius * (1 - (alt_rad_world / (np.pi / 2.0)))

        if r > max_radius + 1e-6 or r < -1e-6: continue

        circle = plt.Circle((center_x, center_y), r, color='white', linestyle=':', linewidth=0.5, fill=False, alpha=0.6)
        ax.add_patch(circle)
        ax.text(center_x, center_y + r, f"{alt_deg}°", color='white', fontsize=8, va='top', ha='center', alpha=0.6)

    for az_deg in range(0, 360, step_deg):
        az_rad_world = np.radians(az_deg)
        # Project horizon points to get angles in image
        x_end = center_x + max_radius * np.sin(-az_rad_world) # Using -az for image X
        y_end = center_y - max_radius * np.cos(-az_rad_world) # Using -sin(-az) for image Y

        ax.plot([center_x, x_end], [center_y, y_end], color='white', linestyle=':', linewidth=0.5, alpha=0.6)
        # Place azimuth labels near the edge
        r_text = max_radius * 0.95
        text_x = center_x + r_text * np.sin(-az_rad_world)
        text_y = center_y - r_text * np.cos(-az_rad_world)
        if alt_deg == step_deg: # Only label once per azimuth line
            ax.text(text_x, text_y, f"{az_deg}°", color='white', fontsize=8, ha='center', va='center', alpha=0.6)

def draw_cardinal_lines_panorama(ax, width=360, height=90):
    # This function draws lines based on azimuth in the panorama's coordinate system (0-360).
    # It adds the camera_heading_deg to the cardinal azimuths.
    # This assumes the panorama's 0-360 azimuth maps to World azimuth + camera_heading.
    # This mapping is likely specific to how fisheye_to_panorama works.
    # Let's keep it as is for the original plotting intent.
    camera_heading_deg = np.degrees(camera_heading) # Assuming global camera_heading for true orientation
    cardinal_azimuths = { # Azimuths relative to NORTH (0)
        "N": 0,
        "E": 90,
        "S": 180,
        "W": 270
    }
    # Project these World Azimuths into the panorama's Azimuth space
    # This assumes the panorama's x-axis (0-360) maps directly to World Azimuth + camera_heading
    # If panorama x=0 is camera forward, then panorama az = World Az - camera_heading.
    # The original code adds camera_heading, suggesting panorama az = World Az + camera_heading?
    # Let's assume the original logic's intent and keep the formula.
    for label, az_deg_world in cardinal_azimuths.items():
        az_deg_panorama = (az_deg_world + camera_heading_deg) % 360
        ax.axvline(x=az_deg_panorama, color='white', linestyle='--', alpha=0.5)
        # text_y_offset = 5 if az_deg_panorama < 180 else -5
        text_y_offset = -15
        ax.text(az_deg_panorama, height / 2 + text_y_offset, label, color='white', fontsize=10,
                ha='center', va='top', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.3, boxstyle='round,pad=0.2'))

def draw_world_axes_panorama(ax, length_deg=25):
    """
    Trace les axes X, Y, Z du repère monde en partant du centre (az=180°, alt=45°) comme vecteurs directionnels.
    """
    origin_az, origin_alt = 180.0, 45.0  # centre de la projection

    axes = {
        "X_W (N)": np.array([0.0, 1.0, 0.0]),
        "Y_W (E)": np.array([1.0, 0.0, 0.0]),
        "Z_W (Up)": np.array([0.0, 0.0, 1.0])
    }

    colors = {
        "X_W (N)": 'blue',
        "Y_W (E)": 'green',
        "Z_W (Up)": 'red'
    }

    for label, vec in axes.items():
        vec = vec.astype(np.float64)
        vec /= np.linalg.norm(vec)

        # direction du vecteur
        alt_rad = np.arcsin(vec[2])
        az_rad = np.arctan2(vec[1], vec[0]) % (2 * np.pi)
        alt_deg = np.degrees(alt_rad)
        az_deg = np.degrees(az_rad)

        # calcul du déplacement directionnel depuis le centre
        delta_az = (az_deg - origin_az + 360) % 360
        if delta_az > 180:
            delta_az -= 360  # conserver dans [-180, 180]
        delta_alt = alt_deg - origin_alt

        ax.arrow(origin_az, origin_alt, delta_az * length_deg / 90, delta_alt * length_deg / 90,
                 color=colors[label], head_width=2, length_includes_head=True)
        ax.text(origin_az + delta_az * length_deg / 90,
                origin_alt + delta_alt * length_deg / 90 + 1,
                label, color=colors[label], fontsize=9, ha='center')
        
# Simulation/Data Parameters
RESOLUTION = 90 # Resolution of irradiance map grid 5<60>90
RESOLUTION_SPECTRAL = 10 # Resolution of spectral data in nm
TIME_STEP  = 30  # Time step in minutes

# === PARAMÈTRES ===

# Chargement des paramètres JSON
def load_parameters(filepath):
    with open(filepath, "r") as file:
        return json.load(file)

parameters = load_parameters("data/obsbr.json")

# Paramètres de la caméra
latitude = parameters["observer"]["latitude"]
longitude = parameters["observer"]["longitude"]
altitude = parameters["observer"]["altitude"]

camera_data = parameters["camera"]
image_width, image_height = camera_data["resolution"]
IMAGE_WIDTH = image_width
IMAGE_HEIGHT = image_height

location = EarthLocation.from_geodetic(lat=latitude * u.deg, lon=longitude * u.deg, height=altitude * u.m)

# Date de capture
timestamp = datetime(2022, 7, 27, 12, 29, 17)
date = datetime(2022, 7, 27)
start_hour = 4
end_hour = 19
n_steps = ((end_hour - start_hour) * 60 // TIME_STEP)  
print(f"Number of images to generate: {n_steps}")

# Image de fond vide (ciel noir)
image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8) # Create a blank image for plotting if skyMap.jpg is missing
loaded_image = cv2.imread("data/skyMapAnonym.jpg")    # Anonym
if loaded_image is None:
    print("Warning: Utilisation d'une image vide pour la visualisation.")
else:
        image = loaded_image # Use the loaded image if successful
        image_height, image_width = loaded_image.shape[:2]

# orientation de la caméra en RADIANS
camera_heading = np.radians(camera_data.get("camera_heading", 0))  # orientation (YAW - axe Y; 0 pour Nord)
camera_alt = np.radians(camera_data.get("camera_alt", 90)) # Altitude (PITCH - Axe X, 0 pour horizon, 90 pour zenith)
camera_roll = np.radians(camera_data.get("camera_roll", 0)) # Rotation autour du zenith (ROLL - axe Z)

pixel_size_x = camera_data["pixel_size_X"]
focal_length = parameters["optics"]["focal_length"]
fov = 2 * np.degrees(np.arctan((pixel_size_x * image_width) / (2 * focal_length)))
# si fov = 180 la projection panoramique prend toute la largeur de l'image
fov = 180

atm_data = parameters["atmosphere"]

spectral_response_dict = {int(k): v for k, v in camera_data["spectral_response"].items()}
wavelengths_camera = np.array(list(spectral_response_dict.keys()), dtype=int)
min_wavelength = np.min(wavelengths_camera)
max_wavelength = np.max(wavelengths_camera)
nm_samples = int((max_wavelength - min_wavelength) / RESOLUTION_SPECTRAL) + 1
if nm_samples < 2:
    nm_samples = 2  # Assurer au moins deux points pour l'interpolation

camera_response_array = np.array([spectral_response_dict[wl] for wl in wavelengths_camera])
camera_response_array /= np.max(camera_response_array)  # Normalisation

# Définir la plage de longueurs d'onde
new_wavelengths_camera = np.linspace(min(wavelengths_camera), max(wavelengths_camera), nm_samples)
# Interpoler la réponse spectrale existante
camera_response_interpolated = quadratic_interp(wavelengths_camera, camera_response_array, new_wavelengths_camera, extrapolate=True)

# Mettre à jour les longueurs d'onde et les réponses spectrales
wavelengths_interp = new_wavelengths_camera
camera_response_array_interp = camera_response_interpolated

""" # Données AM1.5 (simplifiées)
am15_wavelengths = np.array([300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000])  # Valeurs pour interpolation
am15_wavelengths_spaced = np.linspace(min(am15_wavelengths), max(am15_wavelengths), nm_samples)
am15_wavelengths_interpolated = quadratic_interp(am15_wavelengths, am15_wavelengths, am15_wavelengths_spaced, extrapolate=True)

am15_irradiance = np.array([1.5, 1.65, 1.8, 1.75, 1.6, 1.4, 1.15, 0.9, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01])
am15_irradiance_spaced = np.linspace(min(am15_irradiance), max(am15_irradiance), nm_samples)
am15_irradiance_interpolated = quadratic_interp(am15_wavelengths, am15_irradiance, am15_wavelengths_spaced, extrapolate=True)
am15_irradiance_interpolated = (am15_irradiance_interpolated - np.min(am15_irradiance_interpolated)) / (np.max(am15_irradiance_interpolated) - np.min(am15_irradiance_interpolated))
am15_irradiance_interpolated /= np.max(am15_irradiance_interpolated) # Normalisation relative (intensité) non necessaire avec JET
 """

min_wavelength = np.min(wavelengths_camera)
max_wavelength = np.max(wavelengths_camera)
nm_samples = int((max_wavelength - min_wavelength) / RESOLUTION_SPECTRAL) + 1

if nm_samples < 2:
    nm_samples = 2

am15_wavelengths_interpolated, am15_irradiance_interpolated = interpolate_sky_spectrum_to_camera(parameters, RESOLUTION_SPECTRAL)

# if am15_wavelengths_interpolated is not None and am15_irradiance_interpolated is not None:
#     print("Longueurs d'onde interpolées (JSON, sur plage caméra):", am15_wavelengths_interpolated[:5])
#     print("Irradiance interpolée et normalisée (JSON, sur plage caméra):", am15_irradiance_interpolated[:5])
#     print("Nombre d'échantillons:", len(am15_wavelengths_interpolated))

# Interpolation du spectre solaire AM1.5
# solar_spectrum_am15 = quadratic_interp(am15_irradiance_spaced, am15_irradiance_interpolated, wavelengths_interp)
solar_spectrum_am15 = np.interp(wavelengths_interp, am15_wavelengths_interpolated, am15_irradiance_interpolated)
solar_spectrum_am15 /= np.max(solar_spectrum_am15)  # Normalisation

# print("camera's parameters:")
# print(f"  - camera_heading: {camera_heading}°")
# print(f"  - camera_alt: {camera_alt}°")
# print(f"  - camera_Roll: {camera_roll}°")
camera_alt = (np.pi/2 - np.deg2rad(camera_alt))  # Convertir l'angle d'azimut pour la projection fisheye
# Caméra : matrice de rotation
R_cam_to_world, R_world_to_cam = build_world_to_camera_matrix(
    camera_heading, # np.radians(-37),   #  -37
    camera_alt,     # np.radians(90),    # camera_alt 90
    camera_roll       # np.radians(-63)    # camera_roll -63
)

# === PRÉ-CALCUL DE LA TRAJECTOIRE DU SOLEIL ===
sun_traj_xy = []
# Stocke les composantes x, y, z du Soleil dans le repère caméra
vec_cam_list = []
for i in range(n_steps):
    
    # Display progress
    progress = ((i + 1) / n_steps) * 100
    bar_length = 50
    filled_length = int(bar_length * progress // 100)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    # Added '\r' at the beginning
    sys.stdout.write(f"\rCalculating the Sun\'s positions: [{bar}] {progress:.0f}% ")
    sys.stdout.flush()

    hour = start_hour + (end_hour - start_hour) * i / n_steps
    # if n_steps == 1:
    #     hour = start_hour 
    timestamp = Time(datetime(date.year, date.month, date.day, int(hour), int((hour % 1)*60)), scale='utc')

    sun = get_sun(timestamp).transform_to(AltAz(obstime=timestamp, location=location))

    if sun.alt.deg < -5: # Ignorer si le soleil est trop bas (-5 pour marge)
        continue # Passe à l'itération suivante de la boucle

    vec_world = altaz_to_unit_vector(sun.alt.rad, sun.az.rad)
    vec_cam = R_world_to_cam @ vec_world
    vec_cam_list.append(vec_cam) # Stocke le vecteur du soleil dans le repère caméra

    # Basé sur repère caméra : +X=droite, +Y=haut(bas?), +Z=avant (axe optique)
    alt_cam = np.arcsin(vec_cam[2])
    az_cam = np.arctan2(vec_cam[0], vec_cam[1]) % (2 * np.pi)
    x, y = fisheye_projection_equidistant(alt_cam, az_cam, image_width, image_height, fov, np.eye(3))
    if x is not None and y is not None:
        sun_traj_xy.append((x, y))
# Conversion en array pour plotting
vec_cam_array = np.array(vec_cam_list)
sys.stdout.write('\n') # Ajouter un saut de ligne après la fin de la barre de progression
        
# === SIMULATION ===
frames = []
for i in range(n_steps):

    # Display progress
    progress = ((i + 1) / n_steps) * 100
    bar_length = 50
    filled_length = int(bar_length * progress // 100)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f"\rSimulation Progress: [{bar}] {progress:.0f}% ")
    sys.stdout.flush()

    hour = start_hour + (end_hour - start_hour) * i / n_steps
    timestamp = Time(datetime(date.year, date.month, date.day, int(hour), int((hour % 1)*60)), scale='utc')
    sun = get_sun(timestamp).transform_to(AltAz(obstime=timestamp, location=location))

    sun_vec_world = altaz_to_unit_vector(sun.alt.rad, sun.az.rad)
    sun_vec_cam = R_world_to_cam @ sun_vec_world
    sun_alt_cam = np.arcsin(sun_vec_cam[2])
    sun_az_cam = np.arctan2(sun_vec_cam[0], sun_vec_cam[1]) % (2 * np.pi) # azimuth = np.arctan2(dy, dx) dans W ou (dx, dy) pour N-S, E-W

    # irradiance_map, cos_theta = simulate_irradiance_camera_physical(    # simulate_irradiance_camera_v3( # 
    #     sun_alt_cam, sun_az_cam,
    #     wavelengths_interp,
    #     camera_response_interpolated,
    #     solar_spectrum_am15,
    #     atm_data,
    #     resolution=RESOLUTION
    # )  


    irradiance_map, cos_theta = simulate_irradiance_camera_physical_cam( # simulate_irradiance_camera_physical(    # simulate_irradiance_camera_v3( # 
                sun_alt_cam, sun_az_cam,
                R_world_to_cam,
                fov,
                wavelengths_interp,
                camera_response_interpolated,
                solar_spectrum_am15,
                atm_data,
                resolution=RESOLUTION
                )  
    
    wl, spectrum = get_spectral_irradiance_at_max_point(
        irradiance_map,
        cos_theta,
        wavelengths_interp,
        camera_response_interpolated,
        atm_data,
        solar_spectrum_am15)

# --- Vectorisation de la superposition d'irradiance sur l'image fisheye ---
    center_x, center_y = image_width // 2, image_height // 2
    max_radius = min(center_x, center_y)
    y_indices, x_indices = np.meshgrid(np.arange(image_height), np.arange(image_width), indexing='ij')
    distances_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    mask_within_circle = distances_from_center <= max_radius
    
    altitude_radians = np.zeros_like(distances_from_center)
    altitude_radians[mask_within_circle] = distances_from_center[mask_within_circle] / max_radius * np.radians(fov / 2)

    # azimuth_radians_image = np.arctan2(y_indices - center_y, x_indices - center_x) # Angle dans le plan image CCW depuis +X
    azimuth_radians_image = np.arctan2( x_indices - center_x, y_indices - center_y) # Angle dans le plan image CCW depuis +X (inversion N-S)
    
    # Azimut global (basé sur l'angle image + décalage original)
    # Cet azimut est celui calculé pour le pixel, avant l'inversion N-S
    azimuth_radians_global_original = (azimuth_radians_image + np.pi / 2) % (2 * np.pi) # Utilise 'original'

    # # *** CORRECTION POUR L'INVERSION NORD-SUD (Réflexion par rapport à l'axe Est-Ouest) ***
    # # Appliquer la formule arctan2(sin(alpha), -cos(alpha)) pour échanger N et S tout en gardant E et W.
    sin_alpha_original = np.sin(azimuth_radians_global_original)
    cos_alpha_original = np.cos(azimuth_radians_global_original)
    # # L'angle résultant de arctan2 est dans [-pi, pi]. Le mapper dans [0, 2pi).
    azimuth_for_indexing_ns_swapped_rad = (np.arctan2(sin_alpha_original, cos_alpha_original) + 2 * np.pi) % (2 * np.pi)

    # azimuth_east_west_flipped = (2 * np.pi - azimuth_for_indexing_ns_swapped_rad[mask_within_circle]) % (2 * np.pi)

    # Calcul des indices dans la carte d'irradiance (RESOLUTION x RESOLUTION)
    theta_indices_irradiance = np.clip(np.floor(altitude_radians / np.pi * RESOLUTION), 0, RESOLUTION - 1).astype(int)
    
    # L'indice horizontal (phi) est basé sur l'azimut du panorama (maintenant inversé N-S)
       # L'indice horizontal (phi) est basé sur l'azimut inversé N-S
    # Appliquer le masque pour n'indexer que les points dans le cercle VS azimuth_radians_global_original
    phi_indices_irradiance = np.clip(np.floor(azimuth_for_indexing_ns_swapped_rad[mask_within_circle] / (2 * np.pi) * RESOLUTION), 0, RESOLUTION - 1).astype(int) # Utilise l'azimut inversé N-S
    # phi_indices_irradiance = np.clip(np.floor(azimuth_east_west_flipped[mask_within_circle] / (2 * np.pi) * RESOLUTION), 0, RESOLUTION - 1).astype(int)

    # S'assurer que les indices theta sont aussi masqués
    theta_indices_irradiance_masked = theta_indices_irradiance[mask_within_circle]

    irradiance_overlay_fisheye_vectorized = np.zeros((image_height, image_width))
    irradiance_overlay_fisheye_vectorized[y_indices[mask_within_circle], x_indices[mask_within_circle]] = \
        irradiance_map[theta_indices_irradiance_masked, phi_indices_irradiance] # Utilise les indices corrigés

    # Irradiance fisheye vectorisée → colormap
    irr_fisheye = irradiance_overlay_fisheye_vectorized
    # S'assurer que la normalisation ne prend en compte que les valeurs non nulles si le fond est 0
    irr_min = np.min(irr_fisheye[mask_within_circle]) if np.any(mask_within_circle) else 0
    irr_max = np.max(irr_fisheye[mask_within_circle]) if np.any(mask_within_circle) else 1e-9 # Évite division par zéro
    irr_fisheye_norm = (irr_fisheye - irr_min) / (irr_max - irr_min + 1e-9)
    # Appliquer le masque à la normalisation aussi si nécessaire, ou s'assurer que le fond 0 reste 0
    # Si le fond 0 doit rester noir (value 0), on peut juste cliper après normalisation si min n'est pas 0
    # irr_fisheye_norm = np.clip(irr_fisheye_norm, 0, 1) # Utile si min > 0
    irr_fisheye_uint8 = (irr_fisheye_norm * 255).astype(np.uint8)

    irr_fisheye_colormap = cv2.applyColorMap(irr_fisheye_uint8, cv2.COLORMAP_JET) # Utiliser COLORMAP_JET ou COLORMAP_INFERNO
    irr_fisheye_colormap_rgb = cv2.cvtColor(irr_fisheye_colormap, cv2.COLOR_BGR2RGB)
    # Resize la carte panoramique colorisée à la taille de l'image réelle
    # Note: Resize ici semble bizarre, on devrait resize l'image fisheye colorisée à sa propre taille ?
    # Si irr_fisheye_colormap_rgb a déjà la bonne taille (image_height, image_width), le resize n'est pas nécessaire ou devrait être de (width, height).
    irr_fisheye_resized = irr_fisheye_colormap_rgb # Assumer qu'elle a déjà la bonne taille

    # Superposition de la carte d'irradiance sur l'image fisheye
    blended_fisheye = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0.6, irr_fisheye_resized, 0.4, 0)
    # --- Fin de la vectorisation de la superposition fisheye ---

    # --- Vectorisation de la projection panoramique de l'irradiance ---
    """
        Vérifier calcul echelle avc la RESOLUTION * 90, 360 !
    """
    output_shape_panorama =  ((90 * 1//(90 // RESOLUTION)), (360 * 1//(360 // RESOLUTION)))  # Taille de la carte panoramique (altitude soit 90, azimut soit 360)
    alt_steps_pano, az_steps_pano = output_shape_panorama

    alt_coords_pano = np.linspace((fov/2), 0, alt_steps_pano) # Altitude de +90 (haut) à -90 (bas)
    az_coords_pano = np.linspace(0, 360, az_steps_pano, endpoint=False) # Azimut de 0 à 360 degrés

    az_grid_pano, alt_grid_pano = np.meshgrid(az_coords_pano, alt_coords_pano, indexing='xy')

    # zenith_angle_rad_pano = np.radians(90 - alt_grid_pano) # Angle zénithal
    zenith_angle_rad_pano = np.radians(alt_grid_pano) # Angle zénithal

    # L'azimut global pour ce pixel du panorama (avant inversion)
    azimuth_rad_global_pano_original = np.radians(az_grid_pano)

    # # *** CORRECTION POUR L'INVERSION NORD-SUD (Réflexion par rapport à l'axe Est-Ouest) ***
    # # Appliquer la formule arctan2(sin(alpha), -cos(alpha)) pour l'azimut du panorama.
    sin_alpha_pano_original = np.sin(azimuth_rad_global_pano_original)
    cos_alpha_pano_original = np.cos(azimuth_rad_global_pano_original)
    # # L'angle résultant de arctan2 est dans [-pi, pi]. Le mapper dans [0, 2pi).
    azimuth_for_indexing_ns_swapped_pano_rad = (np.arctan2(sin_alpha_pano_original, -cos_alpha_pano_original) + 2 * np.pi) % (2 * np.pi)

    # Calcul des indices dans la carte d'irradiance (RESOLUTION x RESOLUTION)
    theta_indices_pano = np.clip(np.floor(zenith_angle_rad_pano / np.pi * RESOLUTION), 0, RESOLUTION - 1).astype(int)
    # L'indice horizontal (phi) est basé sur l'azimut du panorama (maintenant inversé N-S)
    phi_indices_pano = np.clip(np.floor(azimuth_for_indexing_ns_swapped_pano_rad / (2 * np.pi) * RESOLUTION), 0, RESOLUTION - 1).astype(int) # Utilise l'azimut inversé N-S
    # phi_indices_pano = np.clip(np.floor(azimuth_rad_global_pano_original / (2 * np.pi) * RESOLUTION), 0, RESOLUTION - 1).astype(int) # Utilise l'azimut inversé N-S

    irradiance_overlay_panorama_vectorized = np.zeros(output_shape_panorama)
    irradiance_overlay_panorama_vectorized = irradiance_map[theta_indices_pano, phi_indices_pano] # Utilise les indices corrigés

    # Irradiance panorama vectorisée vers colormap
    irr_pano = irradiance_overlay_panorama_vectorized
    # S'assurer que la normalisation est correcte (éviter div par zéro, gérer fond 0)
    irr_min_pano = np.min(irr_pano) # np.min sur tableau numpy gère les zéros
    irr_max_pano = np.max(irr_pano)
    irr_pano_norm = (irr_pano - irr_min_pano) / (irr_max_pano - irr_min_pano + 1e-8)
    irr_pano_uint8 = (irr_pano_norm * 255).astype(np.uint8)

    irr_pano_colormap = cv2.applyColorMap(irr_pano_uint8, cv2.COLORMAP_JET) # Utiliser COLORMAP_JET ou COLORMAP_INFERNO
    irr_pano_colormap_rgb = cv2.cvtColor(irr_pano_colormap, cv2.COLOR_BGR2RGB)

    # Resize la carte panoramique colorisée à la taille de l'image réelle
    # Note: Assurez-vous que image.shape est bien la taille du panorama généré si vous superposez.
    # Si vous affichez la carte d'irradiance seule en taille output_shape_panorama, le resize n'est pas nécessaire.
    # Si vous superposez sur une image panorama de taille image.shape, alors le resize est nécessaire.
    # Le resize devrait aller à la taille de l'image sur laquelle vous superposez.
    # Si vous superposez sur une image panorama de taille output_shape_panorama:
    # irr_pano_resized = irr_pano_colormap_rgb # Pas besoin de resize
    # Si vous superposez sur l'image fisheye d'origine (ce qui semble être le cas pour blended_panorama):
    irr_pano_resized = cv2.resize(irr_pano_colormap_rgb, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC) # Redimensionne à la taille de l'image fisheye originale

    # Superposition de la carte d'irradiance sur l'image panoramique
    # Attention: 'image' est l'image fisheye ici. Vous voulez probablement superposer sur l'image *panoramique* que vous avez générée.
    # Assurez-vous d'avoir une variable 'image_panorama_originale' ou similaire si vous ne superposez pas sur la fisheye.
    # Si 'image' est bien l'image *panoramique* de base ici, alors c'est correct.
    # Assumons que 'image' est l'image panoramique de base ici pour la superposition.
    blended_panorama = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0.6, irr_pano_resized, 0.4, 0)
    # --- Fin de la vectorisation de la projection panoramique ---

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 2, 
                            width_ratios=[1, 2],    # Adjusted ratio for better balance
                            height_ratios=[2, 1],   # Two rows
                            wspace=0.1,            # Reduced spacing between plots
                            hspace=0.2,            # No vertical spacing needed for single row
                            left=0.02,             # Reduced left margin
                            right=0.98,            # Increased right margin
                            top=0.95,              # Increased top margin
                            bottom=0.15            # Reduced bottom margin
                            )

    # Ajout du titre général
    fig.suptitle(f"Solar Irradiance", fontsize=16, x=0.5, y=0.98) # Ajuster y pour l'espacement

    # Affichage Fisheye (à gauche)
    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(blended_fisheye)

    draw_fisheye_grid(ax0, image_width, image_height, fov_deg=fov, step_deg=15)  # Tracer la grille fisheye

    # Position du Soleil sur la vue fisheye
    x_sun_fish, y_sun_fish = fisheye_projection_equidistant(sun_alt_cam, sun_az_cam, image_width, image_height, fov, np.eye(3))
    if x_sun_fish is not None and y_sun_fish is not None:
        ax0.plot(x_sun_fish, y_sun_fish, 'yo', markersize=8, label='Soleil')

    # Trajectoire du Soleil sur la vue fisheye
    sun_traj_xy_arr = np.array(sun_traj_xy)
    if sun_traj_xy_arr.size > 0 and sun_traj_xy_arr.ndim == 2 and sun_traj_xy_arr.shape[1] == 2:
        ax0.plot(sun_traj_xy_arr[:, 0], sun_traj_xy_arr[:, 1], color='yellow', linestyle='-.', linewidth=1, label='_Trajectoire')

    # Repères E-Nu et caméra sur la vue fisheye
    ax0.plot(image_width/2, image_height/2, 'wo', markersize=3)
    ax0.text(image_width/2 + 5, image_height/2 - 5, 'Z', color='white', fontsize=8, weight='bold')
    for label, az_deg in zip(['N', 'E', 'S', 'W'], [0, 90, 180, 270]):
        az = np.radians(az_deg)
        alt = np.radians(10)
        x, y = fisheye_projection_equidistant(alt, az, image_width, image_height, fov, R_world_to_cam)
        if x is not None and y is not None:
            ax0.text(x, y, label, color='white', fontsize=8,
                ha='center', va='center', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.3, boxstyle='round,pad=0.2'))

    colors = {'X_cam (East)': 'red', 'Y_cam (North)': 'blue', 'Z_cam (Sigth)': 'green'}
    for vec_cam, label, color in zip(R_world_to_cam.T.T, colors.keys(), colors.values()):
        alt = np.arcsin(vec_cam[2])
        az = np.arctan2(vec_cam[0], vec_cam[1]) % (2 * np.pi)
        x_vec, y_vec = fisheye_projection_equidistant(alt, az, image_width, image_height, fov, np.eye(3))
        if x_vec is not None and y_vec is not None:
            ax0.arrow(image_width/2, image_height/2, (x_vec - image_width/2) * 0.35,( y_vec - image_height/2) * 0.35, color=color, width=.5, head_width=15, label=label)
    cam_z = R_world_to_cam.T[:, 2]
    cam_alt = np.arcsin(cam_z[2])
    cam_az = np.arctan2(cam_z[0], cam_z[1]) % (2 * np.pi)
    x_vis, y_vis = fisheye_projection_equidistant(cam_alt, cam_az, image_width, image_height, fov, np.eye(3))
    if x_vis is not None and y_vis is not None:
        ax0.arrow(image_width/2, image_height/2,(x_vis - image_width/2) * 0.5, (y_vis - image_height/2) * 0.5, color='cyan', width=.5, head_width=15, label='Visée')
    
    ax0.set_title(f"Fisheye View\n{hour:.2f} h TU")
    fig.colorbar(ax0.imshow(blended_fisheye, cmap='jet', alpha=0.35), ax=ax0, label='W/m²')
    ax0.set_aspect('equal', adjustable='box')
    ax0.axis('off')
    
    # Affichage Panoramique (à droite)
    ax1 = plt.subplot(gs[0, 1])
    panorama_image = fisheye_to_panorama(blended_panorama.astype(np.uint8), fov_deg=fov, output_shape=(90, 360), camera_alt_deg=np.degrees(camera_alt))
    ax1.imshow(panorama_image, origin='lower', extent=[0, 360, 0, 90])  #, aspect='auto')
    for alt in range(10, 90, 10):  # chaque 10°
        ax1.axhline(alt, color='white', linestyle=':', linewidth=0.5)
    #    ax1.text(0, alt + 1, f"{alt}°", color='white', fontsize=8, va='bottom')
    for az in range(10, 360, 10):  # chaque 30°
        ax1.axvline(az, color='white', linestyle=':', linewidth=0.5)
    #    ax2.text(az, 1, f"{az}°", color='white', fontsize=8, ha='center', va='bottom')

# --- Tracé de la trajectoire du soleil sur la vue panoramique ---
    sun_traj_altaz_pano = []
    center_x, center_y = image_width // 2, image_height // 2
    max_radius = min(center_x, center_y)
    fov_deg_half = fov / 2

    for x_fish, y_fish in sun_traj_xy:
        dx = x_fish - center_x
        dy = y_fish - center_y
        radius_px = np.sqrt(dx**2 + dy**2)
        if radius_px <= max_radius:
            normalized_radius = radius_px / max_radius
            # Angle depuis le centre du champ de vision (theta) en degrés
            theta_deg = normalized_radius * fov_deg_half 
            # Altitude caméra (conversion de theta en altitude pano selon le mapping convenu)
            # Formule compatible : altitude_pano = 90 - theta + camera_alt
            altitude_cam_deg = 90 - theta_deg - (90 - np.degrees(camera_alt))
            # Azimut caméra (conversion de l'angle image)
            # L'angle de l'image (arctan2(dy, dx)) est directement l'azimut pano (en radians puis converti en degrés 0-360)
            azimuth_rad_cam_image = np.arctan2(dy, dx)
            # Convertir l'angle image en degrés et mapper 0-360
            azimuth_deg_cam_image = np.degrees(azimuth_rad_cam_image)
            azimuth_cam_deg_global = (azimuth_deg_cam_image + 360) % 360 # Mapper [-180, 180] ou similaire à [0, 360)

            sun_traj_altaz_pano.append((altitude_cam_deg, azimuth_cam_deg_global))

    if sun_traj_altaz_pano:
        sun_traj_altaz_pano_arr = np.array(sun_traj_altaz_pano)
        ax1.plot(sun_traj_altaz_pano_arr[:, 1], sun_traj_altaz_pano_arr[:, 0], color='yellow', linestyle='-.', linewidth=1, label='_Trajectoire')
                # --- Ajout de la position actuelle du soleil ---
        if x_sun_fish is not None and y_sun_fish is not None:
            dx_sun = x_sun_fish - center_x
            dy_sun = y_sun_fish - center_y
            radius_sun_px = np.sqrt(dx_sun**2 + dy_sun**2)
            if radius_sun_px <= max_radius:
                normalized_radius_sun = radius_sun_px / max_radius
                # Angle theta pour le soleil
                theta_sun_deg = normalized_radius_sun * fov_deg_half # Correction: utilisez un nom clair

                # Altitude soleil (conversion de theta en altitude pano)
                altitude_sun_deg = 90 - theta_sun_deg - (90 - np.rad2deg(camera_alt))

                # Azimut soleil (conversion de l'angle image)
                azimuth_rad_sun_image = np.arctan2(dy_sun, dx_sun)
                azimuth_deg_sun_image = np.degrees(azimuth_rad_sun_image)
                azimuth_sun_deg_global = (azimuth_deg_sun_image + 360) % 360 # Mapper à [0, 360)

                ax1.plot(azimuth_sun_deg_global, altitude_sun_deg, 'yo', markersize=8, label='Soleil (t)')
     
# --- Fin du tracé de la trajectoire sur la vue panoramique ---

    draw_cardinal_lines_panorama(ax1, width=360, height=90)  # Tracer les lignes cardinales sur la vue panoramique
    draw_world_axes_panorama(ax1, length_deg=25)  # Tracer les axes du repère monde sur la vue panoramique
    
    ax1.set_title(f"Panoramic view\n{hour:.2f} h TU")
    ax1.set_xlabel("Az. (°)")
    ax1.set_ylabel("Alt. (°)")
    ax1.axis('on')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(vec_cam_array[:, 2], label="z_cam", color='red')
    ax2.plot(vec_cam_array[:, 0], label="x_cam", color='blue')
    ax2.plot(vec_cam_array[:, 1], label="y_cam", color='green')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_title("Vec_Cam (Sun)")
    ax2.set_xlabel("Time (samples)")
    ax2.set_ylabel("Values")
    ax2.legend()
    ax2.grid(True)
 
    ax3 = plt.subplot(gs[1, 1]) # Placé en bas à droite
    ax3.grid()
    wl, spectrum = get_spectral_irradiance_at_max_point(irradiance_map, cos_theta, wavelengths_interp, camera_response_interpolated, atm_data, solar_spectrum_am15)
    line1, = ax3.plot(wavelengths_interp, camera_response_interpolated, linestyle="-.", label="Camera Response", color='blue', linewidth=1)
    line2, = ax3.plot(wavelengths_interp, solar_spectrum_am15, linestyle="--", label="Solar Spectrum AM1.5", color='orange', linewidth=1)
    line3, = ax3.plot(wl, spectrum, linestyle=":", label="Simulated Spectrum", color='red', linewidth=2)
    ax3.set_xlim(np.min(wavelengths_interp), np.max(wavelengths_interp))
    ax3.set_ylim(0, 1.1)  # Ajuster les limites de l'axe Y pour la réponse de la caméra
    # Paramètres des axes
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel("Relative Intensity")
    # Titres et grilles
    ax3.set_title("Spectrum Responsivity (normalized)")
    ax3.grid(True, linestyle="-", alpha=0.5)
    # Affichage des légendes
    ax3.legend(handles=[line1, line2, line3], loc='upper right')

    fname = f"frame_{i:02d}.png"
    plt.savefig(fname)
    plt.close()
    frames.append(fname)
        
sys.stdout.write('\n') # Ajouter un saut de ligne après la fin de la barre de progression
print("Simulation completed.")

# === EXPORT GIF ===
images = [imageio.imread(f) for f in frames]
imageio.mimsave(f"{timestamp.strftime('%Y%m%d')}_Sun.gif", images, duration=350)
print("Animation saved as GIF.")
# Nettoyage
for f in frames:
    os.remove(f)
print("Temporary files removed.")
