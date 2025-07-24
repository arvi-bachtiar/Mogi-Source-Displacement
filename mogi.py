import numpy as np

def mogi_source(x0, y0, d, r, resolution=50):
    """
    Generate 3D coordinates of a Mogi source sphere.

    Args:
        x0, y0: Horizontal coordinates of the source center (in meters)
        d: Depth of the source center below surface (positive, in meters)
        r: Radius of the sphere (in meters)
        resolution: Number of divisions along theta and phi for smoothness

    Returns:
        xs, ys, zs: 2D arrays representing the surface of the sphere
    """
    # volume
    volume = 4/3 * np.pi * r**3

    # Sphere Surface
    theta = np.linspace(0, 2 * np.pi, resolution * 2)
    phi = np.linspace(0, np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)

    # Parametric surface coordinates
    xs = r * np.sin(phi) * np.cos(theta) + x0
    ys = r * np.sin(phi) * np.sin(theta) + y0
    zs = r * np.cos(phi) - d  # Depth below surface

    return xs, ys, zs, volume

def mogi_surface(nx, ny, x_start, y_start, dx, dy, x0, y0, d, dV, v=0.25):
    """
    Create surface grid and compute vertical displacement using Mogi model.

    Returns:
        X, Y : 2D meshgrid arrays
        Z_flat : original surface elevation (all zero)
        Z_mogi : displaced surface from Mogi model
    """
    x = np.linspace(x_start, x_start + (nx - 1) * dx, nx)
    y = np.linspace(y_start, y_start + (ny - 1) * dy, ny)
    
    X, Y = np.meshgrid(x, y)

    # Mogi vertical displacement
    R = np.sqrt((X - x0)**2 + (Y - y0)**2 + d**2)
    Z_mogi = (1 - v) * dV * d / (np.pi * R**3)

    return X, Y, Z_mogi

def estimate_mogi_volume_change(Z_mogi, nx, ny, x_start, y_start, dx, dy,
                                x0, y0, d, v=0.25):
    """
    Estimate volume change (dV) from vertical displacement using Mogi model.

    Args:
        Z_mogi : 2D array of vertical displacements (same shape as X, Y)
        nx, ny : number of grid points in x and y
        x_start, y_start : start positions in x and y (meters)
        dx, dy : spacing between grid points (meters)
        x0, y0 : source horizontal position (meters)
        d : source depth (meters, positive downward)
        v : Poisson's ratio

    Returns:
        dV_estimate : estimated volume change (scalar, in m³)
    """
    x = np.linspace(x_start, x_start + (nx - 1) * dx, nx)
    y = np.linspace(y_start, y_start + (ny - 1) * dy, ny)
    X, Y = np.meshgrid(x, y)

    # Compute R (distance to source center)
    R = np.sqrt((X - x0)**2 + (Y - y0)**2 + d**2)

    # Invert the Mogi equation to estimate dV at each point
    dV_grid = Z_mogi * np.pi * R**3 / ((1 - v) * d)

    # Take mean or weighted mean over grid
    dV_estimate = np.mean(dV_grid)  # Or use np.median for robustness

    return dV_estimate

def pressure_change(bulk_modulus, delta_v, initial_v):
    """
    Calculate pressure change resulting from a change in volume, assuming a closed system.
    
    Formula:
        ΔP = -K * (ΔV / V₀)
    
    Assumptions:
        - Linear elastic behavior
        - No mass exchange (closed system)
        - Small deformation
        - Isothermal conditions (optional, unless temperature is separately tracked)
    
    Parameters:
        bulk_modulus (float): Bulk modulus of the medium [Pa]
        delta_v (float): Volume change (ΔV) [m³]
        initial_v (float): Initial volume (V₀) [m³]
    
    Returns:
        float: Pressure change (ΔP) [Pa]
    """
    return - bulk_modulus * (delta_v / initial_v)

def temperature_changes(alpha, delta_rho, initial_rho):
    """
    Estimate temperature change from density change due to thermal expansion,
    assuming no phase change and constant pressure.

    Formula:
        ΔT = - (Δρ) / (ρ₀ * α)

    Assumptions:
        - Density change is caused purely by thermal expansion
        - No mass exchange (closed system)
        - No phase change or chemical reactions
        - Linear thermal expansion
        - Isotropic material

    Parameters:
        alpha (float): Volumetric thermal expansion coefficient [1/K]
        delta_rho (float): Density change (Δρ) [kg/m³]
        initial_rho (float): Initial density (ρ₀) [kg/m³]

    Returns:
        float: Temperature change (ΔT) [K]
    """
    return - delta_rho / (initial_rho * alpha)


