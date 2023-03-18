import random
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def first_second_lab():
    def get_average(table, N, p):
        avarage_arr = [0 for i in range(N)]
        for i in range(N):
            avg = 0
            for j in range(p):
                avg = avg + (table[i][j] / p)
            avarage_arr[i] = avg
        return avarage_arr
    def get_dispers(table, avarage, N, p):
        dispers_arr = [0 for i in range(N)]
        for i in range(N):
            disp = 0
            for j in range(p):
                disp = disp + ((table[i][j] - avarage[i])**2)/(p-1)
            dispers_arr[i] = disp
        return dispers_arr
    def G_krit(dispers,N):
        smax = max(dispers)
        print("smax:", smax)
        sumdisp = 0
        for i in range(N):
            sumdisp = sumdisp + dispers[i]
        Gp = smax/sumdisp
        return Gp
    def disp_vosp(dispers,N):
        sumdisp = 0
        for i in range(N):
            sumdisp= sumdisp + dispers[i]
        disp = sumdisp / N
        return disp
    excel = pd.read_excel('C:\\AllPycharn\\TEE\\data.xlsx')
    X = pd.DataFrame(excel, columns=['X1','X2','X3'])
    Y = pd.DataFrame(excel, columns=['Y1','Y2','Y3'])
    Y_np = np.array(Y)
    print('Исходные данные: \n', Y_np)
    N,p = Y_np.shape #N - кол-во строк, p - кол-во столбцов
    avarage_arr = get_average(Y_np, N, p)
    dispers_arr = get_dispers(Y_np, avarage_arr, N,p)

    print('Среднее значение по строкам:', avarage_arr)
    print('Дисперсия по строкам:', dispers_arr)
    Gp = G_krit(dispers_arr, N)
    if Gp < 0.516:
        print("Gp:",Gp)
        print("Оценки дисперсии во всех опытах однородна")
    dispvp = disp_vosp(dispers_arr,N)
    print("Дисперсия воспроизводимости S^2y:",dispvp)


    print("\n2-ЛАБ.РАБОТА\n")
    n = 3
    sb = dispvp/(N*n)
    print("Дисперсия связанная с ошибками в определении коэффициентов регрессии sb:", sb)

    ####
    print("Оценки коэффициентов уравнения регресси:")

    sumsred = 0
    for i in range(N):
        sumsred = sumsred + avarage_arr[i]
    b0 = sumsred/N

    b1 = (avarage_arr[0] - avarage_arr[1] + avarage_arr[2] - avarage_arr[3] + avarage_arr[4] - avarage_arr[5] +
          avarage_arr[6] - avarage_arr[7]) / N

    b2 = (avarage_arr[0] + avarage_arr[1] - avarage_arr[2] - avarage_arr[3] + avarage_arr[4] + avarage_arr[5] -
          avarage_arr[6] - avarage_arr[7]) / N

    b3 = (avarage_arr[0] + avarage_arr[1] + avarage_arr[2] + avarage_arr[3] - avarage_arr[4] - avarage_arr[5] -
          avarage_arr[6] - avarage_arr[7]) / N
    print("b0:", b0, "\nb1:", b1, "\nb2:", b2, "\nb3:", b3)
    ####

    print("Доверительный интервал для коэффициентов регрессии:")
    deltabi = 2.12 * math.sqrt(sb)
    print("delBi:", deltabi)

    print("Расчетные значения t-критерия:")
    # tp0 = math.fabs(b0) / deltabi
    # tp1 = math.fabs(b1) / deltabi
    # tp2 = math.fabs(b2) / deltabi
    # tp3 = math.fabs(b3) / deltabi
    tp0 = math.fabs(b0) / math.sqrt(sb)
    tp1 = math.fabs(b1) / math.sqrt(sb)
    tp2 = math.fabs(b2) / math.sqrt(sb)
    tp3 = math.fabs(b3) / math.sqrt(sb)
    print("tp0:", tp0, "\ntp1:", tp1, "\ntp2:", tp2, "\ntp2:", tp3)
    t_table = 2.13
    print("Уравнение регресси:\n")
    print("y_reg=",b0," + ",b1,"* x1",b2,"* x2",b3,"* x3")

    y_rash = [0 for i in range(N)]
    y_rash[0] = b0 + b1 + b2 + b3
    y_rash[1] = b0 - b1 + b2 + b3
    y_rash[2] = b0 + b1 - b2 + b3
    y_rash[3] = b0 - b1 - b2 + b3
    y_rash[4] = b0 + b1 + b2 - b3
    y_rash[5] = b0 - b1 + b2 - b3
    y_rash[6] = b0 + b1 - b2 - b3
    y_rash[7] = b0 - b1 - b2 - b3
    print("Среднее:                                :", avarage_arr)
    print("расчетные значения параметра оптимизации:", y_rash)

    k = 3
    disp_adv = 0
    for i in range(N):
        disp_adv = disp_adv + (n * (avarage_arr[i]-y_rash[i])**2)/(N - (k + 1))
    print("Дисперсия адекватности S^2ad:",disp_adv)
    F_rash = disp_adv/dispvp
    F_tabl = 3.01
    if F_rash < F_tabl:
        print("F-критерий расчетное:", F_rash, "<", "F-критерий табличное", F_tabl)
    else:
        print("F-критерий расчетное:", F_rash, ">", "F-критерий табличное", F_tabl)

def three_four_five_lab():


    def get_resault(A,B,C,D,X1,X2,X3,X4):
        E = np.random.randint(-1, 1)
        res = A*X1 + B*X2 + C*X3 + D*X4 + E
        return(res)

    def get_average_by_col(array):
        N,P = array.shape
        res_row = [0 for i in range(N)]
        mean = 0
        for i in range(N):
            for j in range(N):
                mean = mean + array[i][j]
            mean = mean / N
            res_row[i] = round(mean,2)
            mean = 0
        return res_row
    def get_avarage_by_row(array):
        N, P = array.shape
        res_row = [0 for i in range(N)]
        mean = 0
        for j in range(N):
            for i in range(N):
                mean = mean + array[i][j]
            mean = mean / N
            res_row[j] = round(mean, 2)
            mean = 0
        return res_row
    def get_diagramms(X, Y, n):
        X_pow = [0 for i in range(n)]
        X_Y_mult = [0 for i in range(n)]

        for i in range(n):
            X_Y_mult[i] = X[i] * Y[i]
            X_pow[i] = X[i]**2
        sum_X = sum(X)
        sum_X_pow = sum(X_pow)
        sum_average_Y = sum(Y)
        sum_X_Y_mult = sum(X_Y_mult)
        print(sum_X,sum_X_pow,sum_average_Y,sum_X_Y_mult)
        kx = (n*sum_X_Y_mult - sum_X * sum_average_Y)/(n * sum_X_pow - sum_X**2)

        b = (sum_average_Y - kx*sum_X)/n
        print("k:",kx,"b:",b)
        print("n:",n)

        x = np.linspace(1, 5, 100)
        Y_line = kx * x + b

        plt.plot(x,Y_line,color="green")
        plt.plot(X,Y,marker='o', linestyle='None')
        plt.show()
        return kx
    def get_average_E(E, N):
        aver = 0
        for i in range(N):
            aver = aver + E[i]
        aver = aver / N
        return aver
    excel = pd.read_excel('C:\\AllPycharn\\TEE\\table_2.xlsx')
    X = pd.DataFrame(excel).to_numpy()
    print('Матрица размера 25x25: \n', pd.DataFrame(X))
    #X = np.array(X)
    N,P = X.shape
    print("N:", N, "P:", P)

    A = 5//2
    B = 7//2
    C = 7-6
    D = 7-7

    #print(A,B,C,D)
    X1 = 0
    X2 = 0
    X3 = 0
    X4 = 0
    res_arr = [[0] * N for i in range(P)]
    y_arr = [0 for i in range(N)]
    k = 5
    X1_arr = [0 for i in range(N)]
    X2_arr = [0 for i in range(N)]
    X3_arr = [0 for i in range(N)]
    X4_arr = [0 for i in range(N)]

    for i in range(N):
        for j in range(N):
            if X[i][j] == 1:
                X3 = (i // k) + 1
                X4 = (i % k) + 1
                X1 = (j // k) + 1
                X2 = (j % k) + 1
                X1_arr[i] = X1
                X2_arr[i] = X2
                X3_arr[i] = X3
                X4_arr[i] = X4
                res_arr[i][j] = get_resault(A,B,C,D,X1,X2, X3, X4) #random.randint(-10,10)

    print(pd.DataFrame(res_arr))
    print("X1_array:",X1_arr)
    print("X2_array:",X2_arr)
    print("X3_array:",X3_arr)
    print("X4_array:",X4_arr)

    for i in range(N):
        for j in range(N):
            if res_arr[i][j]!=0:
                y_arr[i] = res_arr[i][j]
    X1 = 0
    X2 = 0
    X3 = 0
    X4 = 0
    XOneXTwo_arr = np.zeros((k, k))
    XThreeXFour_arr =  np.zeros((k, k))
    for i in range(N):
        for j in range(N):
            if res_arr[i][j]!=0:
                X1 = (j // k)
                X2 = (j % k)
                XOneXTwo_arr[X1][X2] = res_arr[i][j]
    print("X1_X2:")
    print(XOneXTwo_arr.transpose())
    for i in range(N):
        for j in range(N):
            if res_arr[i][j]!=0:
                X1 = (i // k)
                X2 = (i % k)
                XThreeXFour_arr[X1][X2] = res_arr[i][j]
    print("X3_X4:")
    print(XThreeXFour_arr.transpose())

    # average_row_Xonetwo = np.zeros((k, k))
    # average_col_Xonetwo = np.zeros((k, k))
    # average_col_Xthreefour = np.zeros((k, k))
    # average_row_Xthreefour = np.zeros((k, k))
    average_row_Xonetwo = np.zeros(k)
    average_col_Xonetwo = np.zeros(k)
    average_col_Xthreefour = np.zeros(k)
    average_row_Xthreefour = np.zeros(k)

    average_row_Xonetwo = get_avarage_by_row(XOneXTwo_arr)
    average_col_Xonetwo = get_average_by_col(XOneXTwo_arr)
    average_row_Xthreefour = get_avarage_by_row(XThreeXFour_arr)
    average_col_Xthreefour = get_average_by_col(XThreeXFour_arr)

    print("Average X1_X2 by row:", average_row_Xonetwo)
    print("Average X1_X2 by col:", average_col_Xonetwo)

    print("Average X3_X4 by row:", average_row_Xthreefour)
    print("Average X3_X4 by col:",average_col_Xthreefour)

    X1_f = [i+1 for i in range(k)]

    k2 = get_diagramms(X1_f,average_row_Xonetwo,k)
    k1 = get_diagramms(X1_f,average_col_Xonetwo,k)
    k3 = get_diagramms(X1_f,average_col_Xthreefour,k)
    k4 = get_diagramms(X1_f,average_row_Xthreefour,k)


    print("k1:",k1, "k2:",k2, "k3:", k3, "k4:",k4)

    E = [0 for i in range(N)]
    Y_rash = [0 for i in range(N)]
    for i in range(N):
        E[i] = round(y_arr[i] - X1_arr[i]*k1 - X2_arr[i]*k2 - X3_arr[i]*k3 -X4_arr[i]*k4,2)
    print("Ei:",E)
    E_average = get_average_E(E,N)
    print("E_average:",E_average)
    for i in range(N):
        Y_rash[i] = round(X1_arr[i] * k1 + X2_arr[i] * k2 + X3_arr[i] * k3 + X4_arr[i] * k4 + E_average,2)
    print("y:", y_arr)
    print("Y_rash:",Y_rash)


    Y_Y_rash = [0 for i in range(N)]

    for i in range(N):
        Y_Y_rash[i] = round((y_arr[i] - Y_rash[i])**2, 4)
    print("(Y-Y_rash)**2:",Y_Y_rash)
    Y_Y_rash_sum = sum(Y_Y_rash)

    Y_average = sum(y_arr)/ N
    print("Y_average:",Y_average)
    n_m = 1
    Y_disp = [0 for i in range(N)]
    for i in range(N):
        Y_disp[i] = (y_arr[i] - Y_average)**2/(N-1)

    print("Y_dispers:",Y_disp)

    sum_Y_disp = sum(Y_disp)
    k_elem = 4
    dispers_ad = Y_Y_rash_sum/(N-(4+1))
    disp_vosp = sum_Y_disp/N

    print("dispers_ad:",dispers_ad)
    print("disp_vosproi:",disp_vosp)
    F_rash = dispers_ad/disp_vosp
    F_tabl = 4.35
    print("F_rash:",F_rash)
    print("F_tabl:",F_tabl)
first_second_lab()
three_four_five_lab()
#

