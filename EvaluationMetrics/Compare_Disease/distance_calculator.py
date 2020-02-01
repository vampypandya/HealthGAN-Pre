from scipy.stats import wasserstein_distance, ks_2samp, energy_distance, anderson_ksamp


def frobenium_norm(data1, data2):
    pass


def l2_norm(data1, data2):
    pass


def frechet_inception_distance(data1, data2):
    pass


def t_test(data1, data2):
    pass

def energy_dist(data1, data2):
    # data1 = data1.flatten()
    # data2 = data2.flatten()
    ene = energy_distance(data1, data2)
    print(ene)
    return str(ene)

def ks_test(data1, data2):
    # data1 = data1.flatten()
    # data2 = data2.flatten()
    ks = ks_2samp(data1, data2)
    print(ks)
    return str(ks)


def shapiro_will_test(data1, data2):
    pass


def anderson_darling_test(data1, data2):
    # data1 = data1.flatten()
    # data2 = data2.flatten()
    ander = anderson_ksamp([data1, data2])
    print(ander)
    return str(ander)

def wass_distance(data1, data2):
    # data1 = data1.flatten()
    # data2 = data2.flatten()
    wass = wasserstein_distance(data1, data2)
    print(wass)
    return str(wass)


def epps_singleton_test(data1, data2):
    # data1 = data1.flatten()
    # data2 = data2.flatten()
    epps = epps_singleton_2samp(data1, data2)
    print(epps)
    return str(epps)


def calculateDistance(data1,data2,metrics):
    results = {}
    for metric in metrics:
        if metric == 'Wasserstein Distance':
            results[metric] = wass_distance(data1,data2)
        elif metric == 'Frobenius Norm':
            results[metric] = frobenium_norm(data1, data2)
        elif metric == 'L2 Norm':
            results[metric] = l2_norm(data1, data2)
        elif metric == 'Energy Distance':
            results[metric] = energy_dist(data1, data2)
        elif metric == 'Frechet Inception Distance':
            results[metric] = frechet_inception_distance(data1, data2)
        elif metric == 'Students T-test':
            results[metric] = t_test(data1, data2)
        elif metric == 'KS Test':
            results[metric] = ks_test(data1, data2)
        elif metric == 'Shapiro Wil Test':
            results[metric] = shapiro_will_test(data1, data2)
        elif metric == 'Anderson Darling Test':
            results[metric] = anderson_darling_test(data1, data2)
        elif metric == 'Epps Singleton Test':
            results[metric] = epps_singleton_test(data1, data2)

    return results

