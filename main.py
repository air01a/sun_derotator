import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy import units as u
from PIL import Image
import os
from datetime import datetime
import PySimpleGUI as sg
import configparser


height = 0  # Altitude de l'observateur en mètres
config_file = 'config.ini'

def read_config():
    config = configparser.ConfigParser()
    if os.path.exists(config_file):
        config.read(config_file)
    else:
        config['DEFAULT'] = {'Longitude': '3.130800', 'Latitude': '50.669399', 'Directory': '.'}
    return config

def write_config(config):
    with open(config_file, 'w') as configfile:
        config.write(configfile)


def calculate_rotation_speed(time_utc, location, lat, lon):
    time = Time(time_utc)
    time.location=location
    sun = get_sun(time)
    altaz = AltAz(obstime=time, location=location)
    sun_altaz = sun.transform_to(altaz)
    rotation = (np.cos(lat * u.deg)  * np.cos(float(sun_altaz.az.radian))) / np.cos(float(sun_altaz.alt.radian)) *15.0410
    return rotation #*180/np.pi

def rotate_image(image_path, rotation_angle):
    image = Image.open(image_path)
    rotated_image = image.rotate(rotation_angle, resample=Image.BICUBIC, expand=True)
    rotated_image.save(image_path.replace(".bmp", "_rotated.bmp"))

def run(path, lat, lon, writelog):
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)
    first_rotation = -1
    files_dates=[]
    for file in os.listdir(path):
        if file.endswith('.bmp'):
            try:
                date_hour_str = file.split('.')[0].split('_')[0]
                date_hour = datetime.strptime(date_hour_str, '%Y-%m-%d-%H%M')
                date_utc_str = date_hour.strftime('%Y-%m-%dT%H:%M:00')

                files_dates.append((file, date_hour, date_utc_str))
            except ValueError:
                writelog(f'Le fichier {file} ne correspond pas au format attendu.')
    
    file_path,initial_time_utc, time_str = files_dates.pop(0)
    rotation_angle_speed = calculate_rotation_speed(initial_time_utc, location, lat, lon)

    for file_path,time_utc, time_str in files_dates:
        writelog("++++++++ Start",colors='white on green')
        rotation_angle=(time_utc-initial_time_utc).total_seconds()/3600 * rotation_angle_speed
        
        rotate_image(path+'/'+file_path, -rotation_angle)
        writelog(f"Applied a rotation of {rotation_angle:.2f} degrees to {file_path}")

    writelog("Rotation correction applied to all images.")


config = read_config()
lon = config['DEFAULT'].get('Longitude', '')
lat = config['DEFAULT'].get('Latitude', '')
default_directory = config['DEFAULT'].get('Directory', '')

layout = [
    [sg.Text('Longitude'), sg.InputText(lon, key='Longitude')],
    [sg.Text('Latitude'), sg.InputText(lat, key='Latitude')],
    [sg.Text('Directory'), sg.InputText(default_directory, key='Directory'), sg.FolderBrowse()],
    [sg.MLine(size=(100,20), write_only=True, expand_x=True, expand_y=True,key='-LOG-', reroute_cprint=True)],
    [sg.Button('Run'), sg.Button('Exit')]
]

window = sg.Window('Configuration', layout)

while True:
    event, values = window.read()
    
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Run':
        longitude = values['Longitude']
        latitude = values['Latitude']
        directory = values['Directory']
        
        config['DEFAULT']['Longitude'] = longitude
        config['DEFAULT']['Latitude'] = latitude
        config['DEFAULT']['Directory'] = directory
        write_config(config)
        
        run(directory, float(latitude), float(longitude),sg.cprint)


window.close()

