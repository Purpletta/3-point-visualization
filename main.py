import math
import numpy as np
from scipy import spatial
from stl import mesh
from myplot import plot_mesh
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get_point_coordinates(self):
        return (self.x, self.y, self.z)


class Vector3:
    def __init__(self, point_one, point_two):
        self.x = point_two.x - point_one.x
        self.y = point_two.y - point_one.y
        self.z = point_two.z - point_one.z

    def get_vector_coordinates(self):
        return (self.x, self.y, self.z)

    def get_vector_length(self):
        return math.sqrt(sum([x ** 2 for x in [self.x, self.y, self.z]]))


def get_scalar_product(vector_one, vector_two):
    return sum([x * y for x in vector_one.get_vector_coordinates() for y in vector_two.get_vector_coordinates()])


def product_to_const(vector, const):
    vector.x *= const
    vector.y *= const
    vector.z *= const
    return vector


def add_vector_to_point(point, vector):
    return Point(point.x + vector.x, point.y + vector.y, point.z + vector.z)


def get_vector_product(start_point, vector_one, vector_two):
    coords_vector_one = vector_one.get_vector_coordinates()
    coords_vector_two = vector_two.get_vector_coordinates()
    return Vector3(Point(0, 0, 0),
                   Point(*[coords_vector_one[1] * coords_vector_two[2] - coords_vector_one[2] * coords_vector_two[1],
                           coords_vector_one[2] * coords_vector_two[0] - coords_vector_one[0] * coords_vector_two[2],
                           coords_vector_one[0] * coords_vector_two[1] - coords_vector_one[1] * coords_vector_two[0]]))


def get_another_side_square_vector(start_point, vector_one, vector_two):
    new_vector = get_vector_product(start_point, get_vector_product(start_point, vector_one, vector_two), vector_one)
    new_vector = product_to_const(new_vector, vector_one.get_vector_length() / new_vector.get_vector_length())
    return new_vector


def get_equation_of_the_plane(point, vector_one, vector_two):
    # http://www.mathprofi.ru/uravnenie_ploskosti.html
    coord_x = vector_one.y * vector_two.z - vector_one.z * vector_two.y
    coord_y = vector_one.z * vector_two.x - vector_one.x * vector_two.z
    coord_z = vector_one.x * vector_two.y - vector_one.y * vector_two.x
    coord_reserved = -point.x * coord_x - point.y * coord_y - point.z * coord_z
    return [coord_x, coord_y, coord_z, coord_reserved]


def get_top_vertex(coords_plane, start_point, length):
    # https://matworld.ru/analytic-geometry/prjamaja-ploskost-online.php
    time_point = Point(coords_plane[0] + start_point.x,
                       coords_plane[1] + start_point.y,
                       coords_plane[2] + start_point.z)
    time_vector = Vector3(start_point, time_point)
    vector = product_to_const(time_vector, length / time_vector.get_vector_length())
    return add_vector_to_point(start_point, vector)


def get_projection_to_xoy(figure):
    return [[point.x, point.y] for point in figure]


def get_coords_from_user(point):
    print(f'Введите коордитаны точки {point}: ')
    return (int(input("Введите x: ")), int(input("Введите y: ")), int(input("Введите z: ")))


def create_cube(A, B, C):
    # Создание в основании квадрата
    vector_AB = Vector3(A, B)
    vector_AC = Vector3(A, C)
    Length = vector_AB.get_vector_length()

    searched_vector_1 = get_another_side_square_vector(A, vector_AB, vector_AC)
    D = add_vector_to_point(A, searched_vector_1)
    vector_AD = Vector3(A, D)

    E = add_vector_to_point(B, vector_AD)
    # Конец фигуры

    coords_plane = get_equation_of_the_plane(A, vector_AB, vector_AD)

    F = get_top_vertex(coords_plane, A, Length)

    G = get_top_vertex(coords_plane, B, Length)

    P = get_top_vertex(coords_plane, D, Length)

    K = get_top_vertex(coords_plane, E, Length)
    return [A, B, D, E, F, G, P, K]


def create_the_quadrangular_pyramid(A, B, C):
    # Создание в основании квадрата
    vector_AB = Vector3(A, B)
    vector_AC = Vector3(A, C)
    Length = vector_AB.get_vector_length()

    searched_vector_1 = get_another_side_square_vector(A, vector_AB, vector_AC)
    D = add_vector_to_point(A, searched_vector_1)
    vector_AD = Vector3(A, D)

    E = add_vector_to_point(B, vector_AD)
    # Конец фигуры

    coords_plane = get_equation_of_the_plane(A, vector_AB, vector_AD)
    half_AB = add_vector_to_point(A, product_to_const(vector_AB, 0.5))
    central_point = add_vector_to_point(half_AB, product_to_const(vector_AD, 0.5))

    # так как из теоремы пифагора высота будет равна = сторона / 2
    S = get_top_vertex(coords_plane, central_point, Length / math.sqrt(2))
    return [A, B, D, E, S]


def create_tetrahedron(A, B, C):
    # Создание в основании правильного треугольника
    vector_AB = Vector3(A, B)
    vector_AC = Vector3(A, C)
    Length = vector_AB.get_vector_length()

    searched_vector_1 = get_another_side_square_vector(A, vector_AB, vector_AC)
    D = add_vector_to_point(A, searched_vector_1)
    vector_AD = Vector3(A, D)
    coords_plane = get_equation_of_the_plane(A, vector_AB, vector_AD)

    # из формулы высота треугольника равна корень из 3 * сторону * 0.5
    vector_height = product_to_const(vector_AD, math.sqrt(3) / 2)

    half_AB = add_vector_to_point(A, product_to_const(vector_AB, 0.5))
    E = add_vector_to_point(half_AB, vector_height)
    # Конец фигуры

    # по свойству медиан: делятся в точке пересения 2:1
    central_point = add_vector_to_point(half_AB, product_to_const(vector_height, 1 / 3))

    # по теореме пифагора эта высота равна =
    # sqrt((высота бокового треугольника = h * sqrt(3) / 2) ** 2 - (1/3 высоты основания) ** 2)
    S = get_top_vertex(coords_plane, central_point, Length * math.sqrt(2 / 3))
    return [A, B, E, S]


def get_list_of_vertex_cube(cube):
    return [cube[i].get_point_coordinates() for i in range(8)]


def print_tetrahedron_coordinates(tetrahedron):
    name_vertex = ["A", "B", "C", "S"]
    for i in range(4):
        print(f"{name_vertex[i]} = {tetrahedron[i].get_point_coordinates()}")


def print_the_quadrangular_pyramid_coordinates(pyramid):
    name_vertex = ["A", "B", "C", "D", "S"]
    for i in range(5):
        print(f"{name_vertex[i]} = {pyramid[i].get_point_coordinates()}")


def print_plane_coordinates(plane):
    print(f"Plane: {plane[0]}x + {plane[1]}y + {plane[2]}z + {plane[3]} = 0")


def print_cube_coordinates(cube):
    name_vertex = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for i in range(8):
        print(f"{name_vertex[i]} = {cube[i].get_point_coordinates()}")


def draw_figure(figure, name):
    vertices = np.array([x.get_point_coordinates() for x in figure])
    # plot_verticles(vertices=vertices, isosurf=False)

    hull = spatial.ConvexHull(vertices)
    faces = hull.simplices

    myramid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            myramid_mesh.vectors[i][j] = vertices[f[j], :]
    plot_mesh(myramid_mesh)
    myramid_mesh.save(f'{name}.stl')


# Алгоритм Джарвиса для поиска Минимальной Выпуклой Области(МВО)
def jarvismarch(A):
    def rotate(A, B, C):
        return (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])

    n = len(A)
    P = [i for i in range(n)]
    # start point
    for i in range(1, n):
        if A[P[i]][0] < A[P[0]][0]:
            P[i], P[0] = P[0], P[i]
    H = [P[0]]
    del P[0]
    P.append(H[0])
    while True:
        right = 0
        for i in range(1, len(P)):
            if rotate(A[H[-1]], A[P[right]], A[P[i]]) < 0:
                right = i
        if P[right] == H[0]:
            break
        else:
            H.append(P[right])
            del P[right]
    return [[A[x][0] for x in H] + [A[H[0]][0]], [A[x][1] for x in H] + [A[H[0]][1]]]


def draw_projection(figure):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(*jarvismarch(get_projection_to_xoy(figure)))

    plt.show()


def testing_cube(name="test"):
    print(f"Координата определяющей точки -> {p3.get_point_coordinates()}. "
          f"Ниже перечисленны все коррдинаты точек куба в пространстве: \n")
    print_cube_coordinates(create_cube(p1, p2, p3))

    draw_figure(create_cube(p1, p2, p3), name)
    draw_projection(create_cube(p1, p2, p3))


def testing_the_quadrangular_pyramid(name="test"):
    print("Ниже перечисленны все коррдинаты четырёхугольной пирамиды в пространстве: \n")
    print_the_quadrangular_pyramid_coordinates(create_the_quadrangular_pyramid(p1, p2, p3))

    draw_figure(create_the_quadrangular_pyramid(p1, p2, p3), name)
    draw_projection(create_the_quadrangular_pyramid(p1, p2, p3))


def testing_tetrahedron(name="test"):
    print("Ниже перечисленны все коррдинаты точек тэтраэдра в пространстве: \n")
    print_tetrahedron_coordinates(create_tetrahedron(p1, p2, p3))

    draw_figure(create_tetrahedron(p1, p2, p3), name)
    draw_projection(create_tetrahedron(p1, p2, p3))


def get_random_coordinates():
    return np.random.randint(-15, 15, 3)


def is_points_on_line(p1, p2, p3):
    def calculate_the_2x2_determinant(x1, y1, x2, y2):
        return x1 * y2 - x2 * y1

    for x in [("x", "y"), ("y", "z"), ("x", "z")]:
        if eval(f'calculate_the_2x2_determinant(int(p2.{x[0]}) - int(p1.{x[0]}), int(p2.{x[1]}) - int(p1.{x[1]}),'
                f'int(p3.{x[0]}) - int(p1.{x[0]}), int(p3.{x[1]}) - int(p1.{x[1]})) == 0'):
            return True
    return False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    p1 = p2 = p3 = 0
    ans = input("Хотите ли вы ввести координаты вручную или же использовать случайные координаты? (да, нет) -> ")
    if ans == "да":
        p1 = Point(*get_coords_from_user("A"))
        print(p1.get_point_coordinates())
        p2 = Point(*get_coords_from_user("B"))
        p3 = Point(*get_coords_from_user("C"))
    else:
        p1 = Point(*get_random_coordinates())
        p2 = Point(*get_random_coordinates())
        p3 = Point(*get_random_coordinates())
    if not is_points_on_line(p1, p2, p3):
        testing_cube("Cube")
        testing_tetrahedron("Tetrahedron")
        testing_the_quadrangular_pyramid("QudrangularPyramid")
    else:
        print("Все 3 точки находятся на 1 линии, введите иные значения координат")
