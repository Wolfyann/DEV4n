"""
    Test script for the Sun Irradiance simulation.
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
    simulate_irradiance_camera_physical_cam,
    fisheye_to_panorama,
    get_spectral_irradiance_at_max_point
)

def load_parameters(filepath):
    with open(filepath, "r") as file:
        return json.load(file)

def interpolate_camera_response(camera_data, resolution_spectral):
    spectral_response_dict = {int(k): v for k, v in camera_data["spectral_response"].items()}
    wavelengths_camera = np.array(list(spectral_response_dict.keys()), dtype=int)
    min_wavelength = np.min(wavelengths_camera)
    max_wavelength = np.max(wavelengths_camera)
    nm_samples = int((max_wavelength - min_wavelength) / resolution_spectral) + 1
    nm_samples = max(nm_samples, 2)
    camera_response_array = np.array([spectral_response_dict[wl] for wl in wavelengths_camera])
    camera_response_array /= np.max(camera_response_array)
    new_wavelengths_camera = np.linspace(min_wavelength, max_wavelength, nm_samples)
    camera_response_interpolated = quadratic_interp(wavelengths_camera, camera_response_array, new_wavelengths_camera, extrapolate=True)
    return new_wavelengths_camera, camera_response_interpolated

def interpolate_sky_spectrum_to_camera(parameters, resolution_spectral):
    """
        Interpolates the sky spectrum from the data loaded in 'parameters'
        over the wavelength range and with the number of samples
        defined by the camera data.

        Args:
            parameters (dict): Dictionary containing the data loaded from the JSON file.
            resolution_spectral (float): Spectral resolution of the camera.

        Returns:
            tuple: A tuple containing the interpolated wavelengths (np.ndarray)
                and the interpolated and normalized irradiance (np.ndarray),
                or (None, None) in case of error.
    """
    sky_data = parameters.get("sky_model")
    if not sky_data or not sky_data.get("spectrum"):
        print("rror: Key 'sky_model' or 'sky_model.spectrum' is missing in parameters.")
        return None, None
    spectrum_data = sky_data["spectrum"]
    wavelengths_json = np.array([int(k) for k in spectrum_data.keys()])
    irradiance_json = np.array(list(spectrum_data.values()))
    sort_indices_sky = np.argsort(wavelengths_json)
    wavelengths_json = wavelengths_json[sort_indices_sky]
    irradiance_json = irradiance_json[sort_indices_sky]
    camera_data = parameters.get("camera")
    if not camera_data or not camera_data.get("spectral_response"):
        print("Error: Key 'camera' or 'camera.spectral_response' is missing in parameters.")
        return None, None
    spectral_response_dict = {int(k): v for k, v in camera_data["spectral_response"].items()}
    wavelengths_camera = np.array(list(spectral_response_dict.keys()), dtype=int)
    min_wavelength_camera = np.min(wavelengths_camera)
    max_wavelength_camera = np.max(wavelengths_camera)
    nm_samples = int((max_wavelength_camera - min_wavelength_camera) / resolution_spectral) + 1
    nm_samples = max(nm_samples, 2)
    wavelengths_spaced = np.linspace(min_wavelength_camera, max_wavelength_camera, nm_samples)
    irradiance_interpolated = quadratic_interp(wavelengths_json, irradiance_json, wavelengths_spaced, extrapolate=True)
    irradiance_interpolated_normalized = (irradiance_interpolated - np.min(irradiance_interpolated)) / (np.max(irradiance_interpolated) - np.min(irradiance_interpolated))
    irradiance_interpolated_normalized /= np.max(irradiance_interpolated_normalized)
    return wavelengths_spaced, irradiance_interpolated_normalized

def compute_sun_trajectory(n_steps, start_hour, end_hour, date, location, R_world_to_cam, image_width, image_height, fov):
    """
    Computes the trajectory of the Sun in the camera frame for a given day and location.

    For each time step, calculates the Sun's position in the sky, transforms it into the camera frame,
    and projects it onto the fisheye image. Returns both the (x, y) trajectory on the image and the
    corresponding Sun direction vectors in the camera frame.

    Args:
        n_steps (int): Number of time steps to sample between start_hour and end_hour.
        start_hour (float): Start time (in hours, UTC).
        end_hour (float): End time (in hours, UTC).
        date (datetime): Date for the simulation.
        location (EarthLocation): Observer's location.
        R_world_to_cam (np.ndarray): 3x3 rotation matrix from world to camera frame.
        image_width (int): Width of the fisheye image.
        image_height (int): Height of the fisheye image.
        fov (float): Field of view of the fisheye lens (degrees).

    Returns:
        tuple: (sun_traj_xy, vec_cam_array)
            sun_traj_xy (np.ndarray): Array of (x, y) positions of the Sun on the fisheye image.
            vec_cam_array (np.ndarray): Array of Sun direction vectors in the camera frame.
    """

    sun_traj_xy = []
    vec_cam_list = []
    for i in range(n_steps):
        progress = ((i + 1) / n_steps) * 100
        bar_length = 50
        filled_length = int(bar_length * progress // 100)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f"\rCalculating the Sun's positions: [{bar}] {progress:.0f}% ")
        sys.stdout.flush()
        hour = start_hour + (end_hour - start_hour) * i / n_steps
        timestamp = Time(datetime(date.year, date.month, date.day, int(hour), int((hour % 1)*60)), scale='utc')
        sun = get_sun(timestamp).transform_to(AltAz(obstime=timestamp, location=location))
        if sun.alt.deg < -5:
            continue
        vec_world = altaz_to_unit_vector(sun.alt.rad, sun.az.rad)
        vec_cam = R_world_to_cam @ vec_world
        vec_cam_list.append(vec_cam)
        alt_cam = np.arcsin(vec_cam[2])
        az_cam = np.arctan2(vec_cam[0], vec_cam[1]) % (2 * np.pi)
        x, y = fisheye_projection_equidistant(alt_cam, az_cam, image_width, image_height, fov, np.eye(3))
        if x is not None and y is not None:
            sun_traj_xy.append((x, y))
    sys.stdout.write('\n')
    return np.array(sun_traj_xy), np.array(vec_cam_list)

def simulate_and_plot(
    n_steps, start_hour, end_hour, date, location, R_world_to_cam, fov, wavelengths_interp,
    camera_response_interpolated, solar_spectrum_am15, atm_data, RESOLUTION, image, image_width, image_height, sun_traj_xy, vec_cam_array, camera_heading, camera_alt
):
    frames = []
    for i in range(n_steps):
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
        sun_az_cam = np.arctan2(sun_vec_cam[0], sun_vec_cam[1]) % (2 * np.pi)
        irradiance_map, cos_theta = simulate_irradiance_camera_physical_cam(
            sun_alt_cam, sun_az_cam, R_world_to_cam, fov, wavelengths_interp,
            camera_response_interpolated, solar_spectrum_am15, atm_data, resolution=RESOLUTION
        )
        wl, spectrum = get_spectral_irradiance_at_max_point(
            irradiance_map, cos_theta, wavelengths_interp, camera_response_interpolated, atm_data, solar_spectrum_am15
        )
        # --- Vectorized overlay of irradiance on the fisheye image ---
        blended_fisheye = overlay_irradiance_on_fisheye(
            image, irradiance_map, image_width, image_height, fov, RESOLUTION
        )
        # --- Vectorized overlay of irradiance on the panoramic image ---
        blended_panorama = overlay_irradiance_on_panorama(
            image, irradiance_map, fov, RESOLUTION
        )
        # --- Save the blended fisheye and panorama images ---
        fname = plot_all(
            i, hour, blended_fisheye, blended_panorama, sun_alt_cam, sun_az_cam,image, image_width, image_height,
            irradiance_map, sun_traj_xy, vec_cam_array, wl, spectrum, wavelengths_interp, camera_response_interpolated, solar_spectrum_am15, fov, camera_heading, camera_alt, RESOLUTION, R_world_to_cam
        )
        frames.append(fname)
    sys.stdout.write('\n')
    return frames

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

        x_end = center_x + max_radius * np.sin(-az_rad_world)
        y_end = center_y - max_radius * np.cos(-az_rad_world)

        ax.plot([center_x, x_end], [center_y, y_end], color='white', linestyle=':', linewidth=0.5, alpha=0.6)
        
        r_text = max_radius * 0.95
        text_x = center_x + r_text * np.sin(-az_rad_world)
        text_y = center_y - r_text * np.cos(-az_rad_world)
        if alt_deg == step_deg:
            ax.text(text_x, text_y, f"{az_deg}°", color='white', fontsize=8, ha='center', va='center', alpha=0.6)

def draw_cardinal_lines_panorama(ax, width=360, height=90, camera_heading=0):

    camera_heading_deg = np.degrees(camera_heading) 
    cardinal_azimuths = { 
        "N": 0,
        "E": 90,
        "S": 180,
        "W": 270
    }
    for label, az_deg_world in cardinal_azimuths.items():
        az_deg_panorama = (az_deg_world + camera_heading_deg) % 360
        ax.axvline(x=az_deg_panorama, color='white', linestyle='--', alpha=0.5)
    
        text_y_offset = -15
        ax.text(az_deg_panorama, height / 2 + text_y_offset, label, color='white', fontsize=10,
                ha='center', va='top', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.3, boxstyle='round,pad=0.2'))

def draw_world_axes_panorama(ax, length_deg=25):
    """
        Draw x, y, z axes in the panorama view.
    """
    origin_az, origin_alt = 180.0, 45.0 

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

        alt_rad = np.arcsin(vec[2])
        az_rad = np.arctan2(vec[1], vec[0]) % (2 * np.pi)
        alt_deg = np.degrees(alt_rad)
        az_deg = np.degrees(az_rad)

        delta_az = (az_deg - origin_az + 360) % 360
        if delta_az > 180:
            delta_az -= 360
        delta_alt = alt_deg - origin_alt

        ax.arrow(origin_az, origin_alt, delta_az * length_deg / 90, delta_alt * length_deg / 90,
                 color=colors[label], head_width=2, length_includes_head=True)
        ax.text(origin_az + delta_az * length_deg / 90,
                origin_alt + delta_alt * length_deg / 90 + 1,
                label, color=colors[label], fontsize=9, ha='center')

def overlay_irradiance_on_fisheye(image, irradiance_map, image_width, image_height, fov, RESOLUTION):

    center_x, center_y = image_width // 2, image_height // 2
    max_radius = min(center_x, center_y)
    y_indices, x_indices = np.meshgrid(np.arange(image_height), np.arange(image_width), indexing='ij')
    distances_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    mask_within_circle = distances_from_center <= max_radius
    
    altitude_radians = np.zeros_like(distances_from_center)
    altitude_radians[mask_within_circle] = distances_from_center[mask_within_circle] / max_radius * np.radians(fov / 2)

    azimuth_radians_image = np.arctan2( x_indices - center_x, y_indices - center_y)

    azimuth_radians_global_original = (azimuth_radians_image + np.pi / 2) % (2 * np.pi)
    sin_alpha_original = np.sin(azimuth_radians_global_original)
    cos_alpha_original = np.cos(azimuth_radians_global_original)

    azimuth_for_indexing_ns_swapped_rad = (np.arctan2(sin_alpha_original, cos_alpha_original) + 2 * np.pi) % (2 * np.pi)

    theta_indices_irradiance = np.clip(np.floor(altitude_radians / np.pi * RESOLUTION), 0, RESOLUTION - 1).astype(int)
    
    phi_indices_irradiance = np.clip(np.floor(azimuth_for_indexing_ns_swapped_rad[mask_within_circle] / (2 * np.pi) * RESOLUTION), 0, RESOLUTION - 1).astype(int) # Utilise l'azimut inversé N-S
  
    theta_indices_irradiance_masked = theta_indices_irradiance[mask_within_circle]

    irradiance_overlay_fisheye_vectorized = np.zeros((image_height, image_width))
    irradiance_overlay_fisheye_vectorized[y_indices[mask_within_circle], x_indices[mask_within_circle]] = \
        irradiance_map[theta_indices_irradiance_masked, phi_indices_irradiance] 

    irr_fisheye = irradiance_overlay_fisheye_vectorized

    irr_min = np.min(irr_fisheye[mask_within_circle]) if np.any(mask_within_circle) else 0
    irr_max = np.max(irr_fisheye[mask_within_circle]) if np.any(mask_within_circle) else 1e-9
    irr_fisheye_norm = (irr_fisheye - irr_min) / (irr_max - irr_min + 1e-9)
    irr_fisheye_uint8 = (irr_fisheye_norm * 255).astype(np.uint8)

    irr_fisheye_colormap = cv2.applyColorMap(irr_fisheye_uint8, cv2.COLORMAP_JET)
    irr_fisheye_colormap_rgb = cv2.cvtColor(irr_fisheye_colormap, cv2.COLOR_BGR2RGB)
    irr_fisheye_resized = irr_fisheye_colormap_rgb

    blended_fisheye = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0.6, irr_fisheye_resized, 0.4, 0)

    return blended_fisheye

def overlay_irradiance_on_panorama(image, irradiance_map, fov, RESOLUTION):
    output_shape_panorama =  ((90 * 1//(90 // RESOLUTION)), (360 * 1//(360 // RESOLUTION)))
    alt_steps_pano, az_steps_pano = output_shape_panorama

    alt_coords_pano = np.linspace((fov/2), 0, alt_steps_pano)
    az_coords_pano = np.linspace(0, 360, az_steps_pano, endpoint=False)
    
    az_grid_pano, alt_grid_pano = np.meshgrid(az_coords_pano, alt_coords_pano, indexing='xy')

    zenith_angle_rad_pano = np.radians(alt_grid_pano)
    
    azimuth_rad_global_pano_original = np.radians(az_grid_pano)

    sin_alpha_pano_original = np.sin(azimuth_rad_global_pano_original)
    cos_alpha_pano_original = np.cos(azimuth_rad_global_pano_original)
    
    azimuth_for_indexing_ns_swapped_pano_rad = (np.arctan2(sin_alpha_pano_original, -cos_alpha_pano_original) + 2 * np.pi) % (2 * np.pi)

    theta_indices_pano = np.clip(np.floor(zenith_angle_rad_pano / np.pi * RESOLUTION), 0, RESOLUTION - 1).astype(int)

    phi_indices_pano = np.clip(np.floor(azimuth_for_indexing_ns_swapped_pano_rad / (2 * np.pi) * RESOLUTION), 0, RESOLUTION - 1).astype(int) # Utilise l'azimut inversé N-S

    irradiance_overlay_panorama_vectorized = np.zeros(output_shape_panorama)
    irradiance_overlay_panorama_vectorized = irradiance_map[theta_indices_pano, phi_indices_pano]

    irr_pano = irradiance_overlay_panorama_vectorized
    irr_min_pano = np.min(irr_pano)
    irr_max_pano = np.max(irr_pano)
    irr_pano_norm = (irr_pano - irr_min_pano) / (irr_max_pano - irr_min_pano + 1e-8)
    irr_pano_uint8 = (irr_pano_norm * 255).astype(np.uint8)

    irr_pano_colormap = cv2.applyColorMap(irr_pano_uint8, cv2.COLORMAP_JET) 
    irr_pano_colormap_rgb = cv2.cvtColor(irr_pano_colormap, cv2.COLOR_BGR2RGB)

    irr_pano_resized = cv2.resize(irr_pano_colormap_rgb, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC) # Redimensionne à la taille de l'image fisheye originale

    blended_panorama = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0.6, irr_pano_resized, 0.4, 0)
    
    return blended_panorama

def plot_all(
    i, hour, blended_fisheye, blended_panorama, sun_alt_cam, sun_az_cam, image, image_width, image_height,
    irradiance_map, sun_traj_xy, vec_cam_array, wl, spectrum, wavelengths_interp, camera_response_interpolated, solar_spectrum_am15, fov=180, camera_heading=0, camera_alt=0, RESOLUTION=90, R_world_to_cam=None
):
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

    fig.suptitle(f"Solar Irradiance", fontsize=16, x=0.5, y=0.98)

    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(blended_fisheye)

    draw_fisheye_grid(ax0, image_width, image_height, fov_deg=fov, step_deg=15) 

    x_sun_fish, y_sun_fish = fisheye_projection_equidistant(sun_alt_cam, sun_az_cam, image_width, image_height, fov, np.eye(3))
    if x_sun_fish is not None and y_sun_fish is not None:
        ax0.plot(x_sun_fish, y_sun_fish, 'yo', markersize=8, label='Soleil')

    max_y_irradiance_map, max_x_irradiance_map = np.unravel_index(np.argmax(irradiance_map), irradiance_map.shape)
    r_irradiance = (min(image.shape[1], image.shape[0]) / 2) * (max_y_irradiance_map / irradiance_map.shape[0])
    phi_irradiance = (2 * np.pi * (max_x_irradiance_map / irradiance_map.shape[1]) + np.pi) % (2 * np.pi)
    measured_x = int(image.shape[1] / 2 + r_irradiance * np.cos(phi_irradiance))
    measured_y = int(image.shape[0] / 2 - r_irradiance * np.sin(phi_irradiance))
    if measured_x is not None and measured_y is not None:
        ax0.plot(measured_x, measured_y, 'ro', markersize=8, label='_Max Irradiance')

    sun_traj_xy_arr = np.array(sun_traj_xy)
    if sun_traj_xy_arr.size > 0 and sun_traj_xy_arr.ndim == 2 and sun_traj_xy_arr.shape[1] == 2:
        ax0.plot(sun_traj_xy_arr[:, 0], sun_traj_xy_arr[:, 1], color='yellow', linestyle='-.', linewidth=1, label='_Trajectoire')

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
    
    ax1 = plt.subplot(gs[0, 1])
    panorama_image = fisheye_to_panorama(blended_panorama.astype(np.uint8), fov_deg=fov, output_shape=(90, 360), camera_alt_deg=np.degrees(camera_alt))
    ax1.imshow(panorama_image, origin='lower', extent=[0, 360, 0, 90])
    for alt in range(10, 90, 10):
        ax1.axhline(alt, color='white', linestyle=':', linewidth=0.5)
    for az in range(10, 360, 10):  # chaque 30°
        ax1.axvline(az, color='white', linestyle=':', linewidth=0.5)

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
            theta_deg = normalized_radius * fov_deg_half 
            altitude_cam_deg = 90 - theta_deg - (90 - np.degrees(camera_alt))

            azimuth_rad_cam_image = np.arctan2(dy, dx)
            azimuth_deg_cam_image = np.degrees(azimuth_rad_cam_image)
            azimuth_cam_deg_global = (azimuth_deg_cam_image + 360) % 360 
            sun_traj_altaz_pano.append((altitude_cam_deg, azimuth_cam_deg_global))

    if sun_traj_altaz_pano:
        sun_traj_altaz_pano_arr = np.array(sun_traj_altaz_pano)
        ax1.plot(sun_traj_altaz_pano_arr[:, 1], sun_traj_altaz_pano_arr[:, 0], color='yellow', linestyle='-.', linewidth=1, label='_Trajectoire')

        if x_sun_fish is not None and y_sun_fish is not None:
            dx_sun = x_sun_fish - center_x
            dy_sun = y_sun_fish - center_y
            radius_sun_px = np.sqrt(dx_sun**2 + dy_sun**2)
            if radius_sun_px <= max_radius:
                normalized_radius_sun = radius_sun_px / max_radius

                theta_sun_deg = normalized_radius_sun * fov_deg_half
                
                altitude_sun_deg = 90 - theta_sun_deg - (90 - np.rad2deg(camera_alt))

                azimuth_rad_sun_image = np.arctan2(dy_sun, dx_sun)
                azimuth_deg_sun_image = np.degrees(azimuth_rad_sun_image)
                azimuth_sun_deg_global = (azimuth_deg_sun_image + 360) % 360

                ax1.plot(azimuth_sun_deg_global, altitude_sun_deg, 'yo', markersize=8, label='Soleil (t)')


    draw_cardinal_lines_panorama(ax1, width=360, height=90, camera_heading=np.degrees(camera_heading))  # Tracer les lignes cardinales sur la vue panoramique
    draw_world_axes_panorama(ax1, length_deg=25) 
    
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
 
    ax3 = plt.subplot(gs[1, 1])
    ax3.grid()

    line1, = ax3.plot(wavelengths_interp, camera_response_interpolated, linestyle="-.", label="Camera Response", color='blue', linewidth=1)
    line2, = ax3.plot(wavelengths_interp, solar_spectrum_am15, linestyle="--", label="Solar Spectrum AM1.5", color='orange', linewidth=1)
    line3, = ax3.plot(wl, spectrum, linestyle=":", label="Simulated Spectrum", color='red', linewidth=2)
    ax3.set_xlim(np.min(wavelengths_interp), np.max(wavelengths_interp))
    ax3.set_ylim(0, 1.1)
    
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel("Relative Intensity")

    ax3.set_title("Spectrum Responsivity (normalized)")
    ax3.grid(True, linestyle="-", alpha=0.5)

    ax3.legend(handles=[line1, line2, line3], loc='upper right')

    fname = f"frame_{i:02d}.png"
    plt.savefig(fname)
    plt.close()
    return fname

def main():
    # === PARAMETERS ===

    TIME_STEP = 30
    RESOLUTION = 90
    RESOLUTION_SPECTRAL = 5

    parameters = load_parameters("data/dev4n.json")
    latitude = parameters["observer"]["latitude"]
    longitude = parameters["observer"]["longitude"]
    altitude = parameters["observer"]["altitude"]
    camera_data = parameters["camera"]
    image_width, image_height = camera_data["resolution"]
    location = EarthLocation.from_geodetic(lat=latitude * u.deg, lon=longitude * u.deg, height=altitude * u.m)
    date = datetime(2022, 7, 27)
    start_hour = 4
    end_hour = 19
 
    n_steps = ((end_hour - start_hour) * 60 // TIME_STEP)

    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    loaded_image = cv2.imread("data/skyMapAnonym.jpg")
    if loaded_image is not None:
        image = loaded_image
        image_height, image_width = loaded_image.shape[:2]
    camera_heading = np.radians(camera_data.get("camera_heading", 0))
    camera_alt = np.radians(camera_data.get("camera_alt", 90))
    camera_alt = (np.pi/2 - np.deg2rad(camera_alt)) 
    camera_roll = np.radians(camera_data.get("camera_roll", 0))
    pixel_size_x = camera_data["pixel_size_X"]
    focal_length = parameters["optics"]["focal_length"]
    
    # fov = 2 * np.degrees(np.arctan((pixel_size_x * image_width) / (2 * focal_length)))
    fov = 180
    
    atm_data = parameters["atmosphere"]
    wavelengths_interp, camera_response_interpolated = interpolate_camera_response(camera_data, RESOLUTION_SPECTRAL)
    am15_wavelengths_interpolated, am15_irradiance_interpolated = interpolate_sky_spectrum_to_camera(parameters, RESOLUTION_SPECTRAL)
    solar_spectrum_am15 = np.interp(wavelengths_interp, am15_wavelengths_interpolated, am15_irradiance_interpolated)
    solar_spectrum_am15 /= np.max(solar_spectrum_am15)

    R_cam_to_world, R_world_to_cam = build_world_to_camera_matrix(
        camera_heading, camera_alt, camera_roll
    )
    sun_traj_xy, vec_cam_array = compute_sun_trajectory(
        n_steps, start_hour, end_hour, date, location, R_world_to_cam, image_width, image_height, fov
    )
    frames = simulate_and_plot(
        n_steps, start_hour, end_hour, date, location, R_world_to_cam, fov, wavelengths_interp,
        camera_response_interpolated, solar_spectrum_am15, atm_data, RESOLUTION, image, image_width, image_height, sun_traj_xy, vec_cam_array, camera_heading, camera_alt
    )
    # === EXPORT ===
    timestamp = datetime.now()
    images = [imageio.imread(f) for f in frames]
    imageio.mimsave(f"{timestamp.strftime('%Y%m%d')}_Sun.gif", images, duration=350)
    print("Animation saved as GIF.")
    for f in frames:
        os.remove(f)
    print("Temporary files removed.")

if __name__ == "__main__":
    main()
    print("Simulation completed.")
    sys.exit(0)