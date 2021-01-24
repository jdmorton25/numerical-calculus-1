import math
import numpy as np
import os

eps = 1e-3

def a_k(x, k):
        return ( (-1) ** k * x ** (2*k + 1) ) / \
                 ( 2 ** (2*k + 1) * math.factorial(2*k + 1) )

def der_a_k(x, k):
    return ( (-1) ** k * (2*k + 1) * x ** (2*k)) / \
             (2 ** (2*k + 1) * math.factorial(2*k + 1) )

def f(x):
	a = a_k(x, 0)
	S = a_k(x, 0)
	k = 0
	while(not(abs(a/S) < eps)):
		a *= a_k(x, k + 1)/a_k(x, k)
		S += a
		k += 1
	return S

def f_three_steps(x):
	a = a_k(x, 0)
	S = a_k(x, 0)
	k = 0
	for i in range(0, 3):
		a *= a_k(x, k + 1)/a_k(x, k)
		S += a
		k += 1
	return S

def der_f(x):
    a = der_a_k(x, 0)
    S = a
    k = 0
    while(not(abs(a/S) < eps)):
        a = a * der_a_k(x, k + 1)/der_a_k(x, k)
        S += a
        k += 1
    return S

def F(x):
    return (960*x**2 - 20*x**4 + x**6/6)/3840

def integral(interval):
    return F(interval[1]) - F(interval[0])

def g(x):
    return x - np.log(np.exp(-x) - 3*x + 14)

def der_g(x):
    return (-2 + np.exp(x)*(-17 + 3*x))/(-1 + np.exp(x)*(-14 + 3*x))

def phi(x):
    return np.log(np.exp(-x) - 3*x + 14)

# ================================================

def copy(mtx):
    m = []
    for i in range(0, mtx.__len__()):
        m.append([])
        for j in range(0, mtx[0].__len__()):
            m[i].append(mtx[i][j])
    return m

def multiply(a, b):
    result = [[0 for y in range(b[0].__len__())] for x in range(a.__len__())]
    for i in range(0, a.__len__()):
        for j in range(0, b[0].__len__()):
            s = 0;
            for k in range(0, a[0].__len__()):
                s += a[i][k]*b[k][j];
            result[i][j] = s;
    return result

def add(a, b):
    result = [[0 for y in range(a[0].__len__())] for x in range(a.__len__())]
    for i in range(0, a.__len__()):
        for j in range(0, a[0].__len__()):
            result[i][j] = a[i][j] + b[i][j];
    return result

def handle(mtx):
    n = mtx.__len__()
    a = [[0 for y in range(n)] for x in range(n)]
    b = [[0 for y in range(1)] for x in range(n)]
    for i in range(0, n):
        for j in range(0, n):
            a[i][j] = mtx[i][j]
        b[i][0] = mtx[i][-1]
    return a, b

def norma(a, b):
    n = a.__len__()
    maxel = 0
    for i in range(0, n):
        elem = abs( (b[i][0] - a[i][0])/b[i][0] )
        if(elem > maxel):
            maxel = elem
    return maxel

def show(arr):
    s = ""
    for i in range(0, arr.__len__()):
        s += f"x{i} = {arr[i]:.12f}"
        if(i != arr.__len__() - 1):
            s += f", "
    return s

# ================================================

def integral_simp(interval, N):
    h = (interval[1] - interval[0]) / N
    m = N // 2
    J = 0
    J += f(interval[0]) + f(interval[1])
    for i in range(1, 2*m):
        if( (i + 1) % 2 == 0):
            J += 4*f(interval[0] + h*i)
        else:
            J += 2*f(interval[0] + h*i)
    J *= h/3
    return J

def integral_rect(interval, N):
    h = (interval[1] - interval[0]) / N
    J = 0
    for n in range(0, N):
        J += f(interval[0] + n*h + h/2)*h
    return J

def integral_trap(interval, N):
    h = (interval[1] - interval[0]) / N
    J = (f(interval[0]) + f(interval[1])) / 2
    for i in range(1, N):
        J += f(interval[0] + i*h)
    J *= h
    return J

def integral_simp_v2(interval, N):
    h = (interval[1] - interval[0]) / N
    m = N // 2
    J = 0
    J += f_three_steps(interval[0]) + f_three_steps(interval[1])
    for i in range(1, 2*m):
        if( (i + 1) % 2 == 0):
            J += 4*f_three_steps(interval[0] + h*i)
        else:
            J += 2*f_three_steps(interval[0] + h*i)
    J *= h/3
    return J

def integral_rect_v2(interval, N):
    h = (interval[1] - interval[0]) / N
    J = 0
    for n in range(0, N):
        J += f_three_steps(interval[0] + n*h + h/2)*h
    return J

def integral_trap_v2(interval, N):
    h = (interval[1] - interval[0]) / N
    J = (f_three_steps(interval[0]) + f_three_steps(interval[1])) / 2
    for i in range(1, N):
        J += f_three_steps(interval[0] + i*h)
    J *= h
    return J

# ================================================

def gauss(mtx):
    n = mtx.__len__()
    m = copy(mtx)
    for k in range(0, n - 1):
        for i in range(k + 1, n):
            div = m[i][k] / m[k][k]
            m[i][n] -= div * m[k][n]
            for j in range(k, n):
                m[i][j] -= div * m[k][j]
    x = [0 for i in range(n)]
    for k in range(n - 1, -1, -1):
        x[k] = (m[k][-1] - sum([m[k][j] * x[j] for j in range(k + 1, n)])) / m[k][k]
    return x

def iterions(mtx):
    n = mtx.__len__()
    a, b = handle(mtx)
    a1, b1 = copy(a), copy(b)
    for i in range(n):
        for j in range(n):
            a1[i][j] = -a[i][j]/a[i][i]
        b1[i][0] = b[i][0]/a[i][i]
        a1[i][i] = 0
    x = copy(b1)
    ns = add(b1, multiply(a1, x))
    while( not( norma(ns, x) < eps ) ):
        x = copy(ns)
        ns = add(b1, multiply(a1, x))
    result = [0 for x in range(n)]
    for i in range(n):
        result[i] = x[i][0]
    return result

def seidel(mtx):
    n = mtx.__len__()
    a, b = handle(mtx)
    x = [[0 for y in range(1)] for x in range(n)]
    converge = False
    while(not(converge)):
        x_new = copy(x)
        for i in range(n):
            s1 = sum(a[i][j] * x_new[j][0] for j in range(i))
            s2 = sum(a[i][j] * x[j][0] for j in range(i + 1, n))
            x_new[i][0] = (b[i][0] - s1 - s2) / a[i][i]
        converge = norma(x, x_new) < eps
        x = x_new
    result = [0 for x in range(n)]
    for i in range(n):
        result[i] = x[i][0]
    return result

# ================================================

def iterions_v2(mtx, epsilon):
    counter = 1
    n = mtx.__len__()
    a, b = handle(mtx)
    a1, b1 = copy(a), copy(b)
    for i in range(n):
        for j in range(n):
            a1[i][j] = -a[i][j]/a[i][i]
        b1[i][0] = b[i][0]/a[i][i]
        a1[i][i] = 0
    x = copy(b1)
    ns = add(b1, multiply(a1, x))
    while( not( norma(ns, x) < epsilon ) ):
        x = copy(ns)
        ns = add(b1, multiply(a1, x))
        counter += 1
    result = [0 for x in range(n)]
    for i in range(n):
        result[i] = x[i][0]
    return result, counter

def seidel_v2(mtx, epsilon):
    counter = 1
    n = mtx.__len__()
    a, b = handle(mtx)
    x = [[0 for y in range(1)] for x in range(n)]
    converge = False
    while(not(converge)):
        x_new = copy(x)
        for i in range(n):
            s1 = sum(a[i][j] * x_new[j][0] for j in range(i))
            s2 = sum(a[i][j] * x[j][0] for j in range(i + 1, n))
            x_new[i][0] = (b[i][0] - s1 - s2) / a[i][i]
        converge = norma(x, x_new) < epsilon
        x = x_new
        counter += 1
    result = [0 for x in range(n)]
    for i in range(n):
        result[i] = x[i][0]
    return result, counter

def relax(mtx, epsilon):
    M = 100
    n = mtx.__len__()
    a, b = handle(mtx)
    test = []
    for q in range(0, M):
        counter = 1
        x = [[0 for y in range(1)] for x in range(n)]
        converge = False
        omega = 1 + 1/M*q
        while(not(converge)):
            x_new = copy(x)
            for i in range(n):
                s1 = sum(a[i][j] * x_new[j][0] for j in range(i))
                s2 = sum(a[i][j] * x[j][0] for j in range(i + 1, n))
                x_new[i][0] = (1 - omega) * x[i][0] + omega * (b[i][0] - s1 - s2) / a[i][i]
            converge = norma(x, x_new) < epsilon
            x = x_new
            counter += 1
            if converge:
                test.append([counter, copy(x), omega])
    test.sort(key=lambda row: row[0], reverse=False)
    
    result = [0 for x in range(n)]
    counter = test[0][0]
    for i in range(n):
        result[i] = test[0][1][i][0]
    omega = test[0][2]
    return result, counter, omega

# ================================================

def dichotomy_method(interval, epsilon):
    counter = 0
    a = interval[0]
    b = interval[1]
    while not(b - a < epsilon):
        c = (a + b) / 2;
        if(g(b) * g(c) < 0):
            a = c;
        else:
            b = c;
        counter+=1
    return (a + b)/2, counter

def newton_method(x0, epsilon):
    counter = 0
    x = x0
    converge = False
    while not(converge):
        x_new = x - f(x)/der_g(x)
        converge = abs(x_new - x) < epsilon
        x = x_new
        counter+=1
    return x, counter

def iter_method(x0, epsilon):
    counter = 0
    x = x0
    converge = False
    while not(converge):
        x_new = phi(x)
        converge = abs(x_new - x) < epsilon
        x = x_new
        counter+=1
    return x, counter

# ================================================


def first():
    x = .5   
    a = a_k(x, 0)
    S = a
    k = 0
    while(not(abs(a/S) < eps)):
        a = a * a_k(x, k + 1)/a_k(x, k)
        S += a
        k += 1
    print(f"The value of S is {S} \nThe value of k is {k}")

def second():
    x = .5
    a = der_a_k(x, 0)
    S = a
    k = 0
    while(not(abs(a/S) < eps)):
        a = a * der_a_k(x, k + 1)/der_a_k(x, k)
        S += a
        k += 1
    print(f"The value of S is {S} \nThe value of k is {k}")

def third():
    start = 0.1
    delta = 0.1
    for i in range(0, 5):
        point = start + i*delta
        print(f"x: {point:.1f} f(x): {f(point):.5f}")

def fourth():
    start = 0.1
    delta = 0.1
    x = []
    y = []
    for i in range(0, 5):
        point = start + i*delta
        x.append(point)
        print(f"x: {point:.1f} f(x): {f(point):.10f}")
        y.append(f(point))
    for i in range (0, 4):
        c0 = y[i] - (y[i + 1] - y[i])/(x[i + 1] - x[i]) * x[i]
        c1 = (y[i + 1] - y[i])/(x[i + 1] - x[i])
        point = x[i] + delta/2
        p = c0 + c1*point
        print(f"x: {point:.2f} p(x): {p:.10f}")

def fifth():
    start = 0.1
    delta = 0.1
    x = []
    y = []
    p = []
    for i in range(0, 5):
        point = start + i*delta
        x.append(point)
        print(f"x: {point:.1f} f(x): {f(point):.10f}")
        y.append(f(point))
    print()
    for i in range (0, 4):
        c0 = y[i] - (y[i + 1] - y[i])/(x[i + 1] - x[i]) * x[i]
        c1 = (y[i + 1] - y[i])/(x[i + 1] - x[i])
        point = x[i] + delta/2
        p.append(c0 + c1*point)
        print(f"x: {point:.2f} p(x): {p[i]:.10f}")
    print()
    for i in range (0, 4):
        print(f"x: {x[i] + delta/2:.2f} z(x): {abs(f(x[i] + delta/2) - p[i]):.10f}")

def sixth():
    start = 0.1
    delta = 0.1
    print("Forward difference: ")
    for i in range(1, 5):
        x = start + i*delta
        f_plus  = (f(x + delta) - f(x))/delta
        print(f"x = {x:.2f}, f_plus = {f_plus}")
    print("\nBackward difference: ")
    for i in range(1, 5):
        x = start + i*delta
        f_minus = (f(x) - f(x - delta))/delta
        print(f"x = {x:.2f}, f_minus = {f_minus}")
    print("\nCentral difference: ")
    for i in range(1, 5):
        x = start + i*delta
        f_zero  = (f(x + delta) - f(x - delta))/(2*delta)
        print(f"x = {x:.2f}, f_zero = {f_zero}")

def seventh():
    start = 0.1
    delta = 0.1
    print("Forward difference: ")
    for i in range(1, 5):
        x = start + i*delta
        f_plus  = (f(x + delta) - f(x))/delta
        print(f"x = {x:.2f}, f_plus = {f_plus:.8f}, z_1(x) = {abs(der_f(x) - f_plus):.8f}")
    print("\nBackward difference: ")
    for i in range(1, 5):
        x = start + i*delta
        f_minus = (f(x) - f(x - delta))/delta
        print(f"x = {x:.2f}, f_minus = {f_minus:.8f}, z_1(x) = {abs(der_f(x) - f_minus):.8f}")
    print("\nCentral difference: ")
    for i in range(1, 5):
        x = start + i*delta
        f_zero  = (f(x + delta) - f(x - delta))/(2*delta)
        print(f"x = {x:.2f}, f_zero = {f_zero:.8f}, z_1(x) = {abs(der_f(x) - f_zero):.8f}")

def eighth():    
    # simpson's integral test
    n = 10
    while( not( abs( ( integral_simp( (1, 2), n ) - integral_simp( (1, 2), 2*n ) ) / integral_simp( (1, 2), 2*n ) ) < eps ) ):
        n *= 2
    print(f"Simpson's formula: J_n(f) is {integral_simp( (1, 2), n )}, n is {n}")
    # rect integral test
    n = 10
    while( not( abs( ( integral_rect( (1, 2), n ) - integral_rect( (1, 2), 2*n ) ) / integral_rect( (1, 2), 2*n ) ) < eps ) ):
        n *= 2
    print(f"Rectangle method: J_n(f) is {integral_rect( (1, 2), n )}, n is {n}")
    # trapezoid integral test
    n = 10
    while( not( abs( ( integral_trap( (1, 2), n ) - integral_trap( (1, 2), 2*n ) ) / integral_trap( (1, 2), 2*n ) ) < eps ) ):
        n *= 2
    print(f"Trapezoid method: J_n(f) is {integral_trap( (1, 2), n )}, n is {n}")

def ninth():
    # integral test
    print(f"Integral value: J(f) is {integral((1, 2))}")
    # simpson's integral test
    n = 10
    while( not( abs( ( integral_simp_v2( (1, 2), n ) - integral_simp_v2( (1, 2), 2*n ) ) / integral_simp_v2( (1, 2), 2*n ) ) < eps ) ):
        n *= 2
    print(f"Simpson's formula: J_n(f) is {integral_simp_v2( (1, 2), n )}, n is {n}")
    print(f"Simpson's formula: R_2n(f) is {abs( ( integral((1, 2)) - integral_simp_v2( (1, 2), 2*n ) ) / integral_simp_v2( (1, 2), 2*n ) )}")
    # rect integral test
    n = 10
    while( not( abs( ( integral_rect_v2( (1, 2), n ) - integral_rect_v2( (1, 2), 2*n ) ) / integral_rect_v2( (1, 2), 2*n ) ) < eps ) ):
        n *= 2
    print(f"Rectangle method: J_n(f) is {integral_rect_v2( (1, 2), n )}, n is {n}")
    print(f"Rectangle method: R_2n(f) is {abs( ( integral((1, 2)) - integral_rect_v2( (1, 2), 2*n ) ) / integral_rect_v2( (1, 2), 2*n ) )}")
    # trapezoid integral test
    n = 10
    while( not( abs( ( integral_trap_v2( (1, 2), n ) - integral_trap_v2( (1, 2), 2*n ) ) / integral_trap_v2( (1, 2), 2*n ) ) < eps ) ):
        n *= 2
    print(f"Trapezoid method: J_n(f) is {integral_trap_v2( (1, 2), n )}, n is {n}")
    print(f"Trapezoid method: R_2n(f) is {abs( ( integral((1, 2)) - integral_trap_v2( (1, 2), 2*n ) ) / integral_trap_v2( (1, 2), 2*n ) )}")

def tenth():
    matrix = [[ 7, .6, .5, 55.1],
              [.6,  6, .4, 42.1],
              [.5, .4,  5, 30.9]]
    print("Solutions:")
    print(f"    Gauss's method    : {show(gauss(matrix))}")
    print(f" Iteratations' method : {show(iterions(matrix))}")
    print(f"    Seidel's method   : {show(seidel(matrix))}")

def eleventh():
    matrix = [[ 7, .6, .5, 55.1],
              [.6,  6, .4, 42.2],
              [.5, .4,  5, 30.9]]

    gauss_sol = gauss(matrix)
    iters_sol, iters_counter = iterions_v2(matrix, eps)
    seidel_sol, seidel_counter = seidel_v2(matrix, eps)
    relax_sol, relax_counter, relax_omega = relax(matrix, eps)

    a, b = handle(matrix)

    g = np.reshape(np.array(gauss_sol), (3, 1)).tolist()
    i = np.reshape(np.array(iters_sol), (3, 1)).tolist()
    s = np.reshape(np.array(seidel_sol), (3, 1)).tolist()
    r = np.reshape(np.array(relax_sol), (3, 1)).tolist()

    print("Gauss's method:")
    print(" Solutions: ")
    print(f" {show(gauss_sol)}")
    print(" Validation: ")
    print(f" b = {np.reshape(multiply(a, g), (3)).tolist()}")

    print("\nIterions's method:")
    print(" Solutions: ")
    print(f" {show(iters_sol)}")
    print(" Counter: ")
    print(f" iters_counter = {iters_counter}")
    print(" Validation: ")
    print(f" b = {np.reshape(multiply(a, i), (3)).tolist()}")

    print("\nSeidel's method:")
    print(" Solutions: ")
    print(f" {show(seidel_sol)}")
    print(" Counter: ")
    print(f" seidel_counter = {seidel_counter}")
    print(" Validation: ")
    print(f" b = {np.reshape(multiply(a, s), (3)).tolist()}")

    print("\nRelaxation's method:")
    print(" Solutions: ")
    print(f" {show(relax_sol)}")
    print(" Counter: ")
    print(f" relax_counter = {relax_counter}")
    print(" Omega: ")
    print(f" relax_omega = {relax_omega}")
    print(" Validation: ")
    print(f" b = {np.reshape(multiply(a, r), (3)).tolist()}")

    
    
    
def twelveth():
    dich_sol, dich_counter = dichotomy_method((0, 3.8), eps)
    print("Dichotomy method:")
    print(f" Solution: x = {dich_sol}")
    print(f" Counter: dich_counter = {dich_counter}")

    new_sol, new_counter = newton_method(1.9, eps)
    print("\nNewton's method:")
    print(f" Solution: x = {new_sol}")
    print(f" Counter: new_counter = {new_counter}")

    iter_sol, iter_counter = iter_method(1.9, eps)
    print("\nIteration method:")
    print(f" Solution: x = {iter_sol}")
    print(f" Counter: iter_counter = {iter_counter}")

def thirteen():
    dich_sol, dich_counter = dichotomy_method((0, 3.8), eps)
    print("Dichotomy method:")
    print(f" Solution: x = {dich_sol}")
    print(f" Counter: dich_counter = {dich_counter}")
    print(f" Error rate: r(x_n) = {f(dich_sol)}")

    new_sol, new_counter = newton_method(1.9, eps)
    print("\nNewton's method:")
    print(f" Solution: x = {new_sol}")
    print(f" Counter: new_counter = {new_counter}")
    print(f" Error rate: r(x_n) = {f(new_sol)}")

    iter_sol, iter_counter = iter_method(1.9, eps)
    print("\nIteration method:")
    print(f" Solution: x = {iter_sol}")
    print(f" Counter: iter_counter = {iter_counter}")
    print(f" Error rate: r(x_n) = {f(iter_sol)}")

def switch(argument):
    switcher = {
        1: "first()",
        2: "second()",
        3: "third()",
        4: "fourth()",
        5: "fifth()",
        6: "sixth()",
        7: "seventh()",
        8: "eighth()",
        9: "ninth()",
        10: "tenth()",
        11: "eleventh()",
        12: "twelveth()",
        13: "thirteen()"
    }
    print("")
    return switcher.get(argument)

while(True):
    inp = ""
    acceptable = False
    while(not(acceptable)):
        print(" 1. Task 1. Item 1 \n" +
              " 2. Task 1. Item 2 \n" +
              " 3. Task 1. Item 3 \n" +
              " 4. Task 1. Item 4 \n" +
              " 5. Task 1. Item 5 \n" +
              " 6. Task 1. Item 6 \n" +
              " 7. Task 1. Item 7 \n" +
              " 8. Task 2. Item 1 \n" +
              " 9. Task 2. Item 2 \n" +
              "10. Task 3. Item 1 \n" +
              "11. Task 3. Item 2 \n" +
              "12. Task 4. Item 1 \n" +
              "13. Task 4. Item 2 ")
        inp = int(input("Enter an item from list: "))
        if(inp > 0 and inp < 14): 
            acceptable = True
    os.system("cls")
    eval(switch(inp))
    print("")
    os.system("pause")
    os.system("cls")
