import numpy as np

def cartesian2polar(x, y):
    r = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0))
    phi = np.arctan2(y, x)
    return r, phi

def polar2cartesian(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

def cartesian2cylindrical(x, y, z):
    r = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0))
    phi = np.arctan2(y, x)
    return r, phi, z

def cylindrical2cartesian(r, phi, z):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z

def cartesian2spherical(x, y, z):
    r = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0) + np.power(z, 2.0))
    phi = np.arctan2(y, x)
    theta = np.arccos(z/r)
    return r, theta, phi

def spherical2cartesian(r, phi, theta):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.cos(theta)
    return x, y, z