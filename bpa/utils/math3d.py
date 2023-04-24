import numpy as np

def get_center_from_triangle_and_radius(a, b, c, r):
    '''
    Given three 3D points and a radius, find the center of the sphere that has the three points on its surface.
    @Algorihtm:
        1. Find the circumcenter of the triangle
        2. Give the circumcenter a perpendicular modification to some height
    @Reference:
        https://gamedev.stackexchange.com/questions/60630/how-do-i-find-the-circumcenter-of-a-triangle-in-3d
    '''
    ac = c - a
    ab = b - a
    abXac = np.cross(ab, ac)
    circumcenter = a + (np.cross(abXac, ab) * np.dot(ac, ac) + np.cross(ac, abXac) * np.dot(ab, ab)) / (2 * np.dot(abXac, abXac))
    normal = abXac / np.linalg.norm(abXac)
    delta = r ** 2 - np.dot(circumcenter-a, circumcenter-a)
    if delta < 0:
        return None
    return circumcenter + normal * np.sqrt(delta)

def project_point_to_plane(p, a, n):
    '''
    Project a point p to a plane (a,n)
    where a is a point on the plane and n is the normal of the plane
    '''
    return p - np.dot(p-a, n) * n

def dihedral_angle(a, b, c, d):
    '''
    A line (a,b) and two points c and d. Calc the dihedral angle (c, (a,b), d)
    Return a rad angle in [0, 2*pi)
    '''
    n = (b-a) / np.linalg.norm(b-a)
    c = project_point_to_plane(c, a, n)
    d = project_point_to_plane(d, a, n)
    # Now a,c,d are on the same plane
    v1 = (c-a) / np.linalg.norm(c-a)
    v2 = (d-a) / np.linalg.norm(d-a)
    angle = np.arccos(np.dot(v1, v2))
    if np.dot(np.cross(v1, v2), n) < 0:
        angle = 2 * np.pi - angle
    return angle
    