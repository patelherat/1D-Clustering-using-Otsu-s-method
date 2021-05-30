import pandas as pd
import statistics
import math
import matplotlib.pyplot as plt

data = pd.read_csv("Abominable_Data_For_Clustering__v205.csv")      # reads csv file


# calculates variance
def var(ages_partitioned):
    if len(ages_partitioned) == 0 or len(ages_partitioned) == 1:
        return 0
    return statistics.variance(ages_partitioned)


def main():
    best_mixed_variance = float('inf')
    best_threshold = 0
    ages = data['Age'].tolist()

    ages2 = []
    for i, age in enumerate(ages):
        ages2.append(math.floor(age / 2) * 2)       # quantize the snowfolks age into bins(of 2 years)

    mvar = []
    ages1 = []
    ages2.sort()                                    # sorting the data

    # otsu's method
    for i, age in enumerate(ages2):
        wt_left = len(ages2[:i+1]) / len(ages2)
        wt_right = len(ages2[i+1:]) / len(ages2)
        var_left = var(ages2[:i+1])
        var_right = var(ages2[i+1:])
        mixed_variance = (wt_left * var_left) + (wt_right * var_right)
        mvar.append(mixed_variance)
        ages1.append(age)

        if mixed_variance <= best_mixed_variance:
            best_mixed_variance = mixed_variance        # stores minimum variance
            best_threshold = age                        # age to separate two clusters

    print("Age used to separate 2 clusters is:", best_threshold)
    print("Minimum mixed variance is:", best_mixed_variance)

    # plots graph for mixed variance for snowfolk's data versus age
    plt.plot(ages1, mvar)
    plt.plot(best_threshold, best_mixed_variance, 'ro')
    plt.ylabel("Mixed Variance")
    plt.xlabel("Age")
    plt.show()

    alphas = [100, 1, 1/5, 1/10, 1/20, 1/25, 1/50, 1/100, 1/1000]   # learning rates
    norm_factor = 100


    for alpha in alphas:
        cost_fun = []
        min_cost_function = float('inf')
        best_age = 0
        for i, age in enumerate(ages1):
            regularisation = alpha * abs(len(ages[:i+1]) - len(ages[i+1:])) / norm_factor
            cost_function = mvar[i] + regularisation
            cost_fun.append(cost_function)
            if cost_function < min_cost_function:
                min_cost_function = cost_function
                best_age = age
                # best_alpha = alpha
        print("Minimized cost function and alpha value and age", min_cost_function, alpha, best_age)
        plt.figure()
        plt.plot(ages1, cost_fun)
        plt.ylabel("cost function")
        plt.xlabel("age")
        plt.show()

    print("Minimized cost function with regularization", min_cost_function)
    # print("Best alpha value is", best_alpha)


if __name__ == '__main__':
    main()
