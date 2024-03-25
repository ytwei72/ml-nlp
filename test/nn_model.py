import math

from common.np import *


def show_cos_sim():
    vec1 = np.array([1, 2, 3, 4, 5])
    vec2 = np.array([3, 4, 5, 6, 7])
    vec3 = np.array([1, 2, 3, 4, 5])
    simi12 = np.sum(vec1 * vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    simi13 = np.sum(vec1 * vec3) / (np.linalg.norm(vec1) * np.linalg.norm(vec3))
    print(simi12, simi13)
    print("np.sum(vec1 * vec2):", np.sum(vec1 * vec2))
    print("np.sum(vec1 * vec3):", np.sum(vec1 * vec3))
    print("np.linalg.norm(vec1):", np.linalg.norm(vec1))
    print("np.linalg.norm(vec2):", np.linalg.norm(vec2))
    print("np.linalg.norm(vec3):", np.linalg.norm(vec3))
    print(math.sqrt(55))


show_cos_sim()
