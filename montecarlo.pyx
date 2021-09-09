import numpy as np

cpdef random_walk_terminal(mu: double, sigma: double, S0: double, nt: int, paths: int):
    """
    This function return the terminal value of a monte carlo simulation
    :param mu: mean | return
    :param sigma: std | risk
    :param S0: inicial price
    :param nt: Number of Time Steps
    :param paths: number of paths to simulate
    :return: número de valores finais dessa série
    """
    mu = mu
    sigma = sigma
    s0 = S0

    Nt = nt
    Np = paths

    cdef double dt = 1/252
    cdef int NtP = Nt +1


    z = np.zeros(shape=(Nt, Np))

    cdef int i
    cdef int j
    for i in range(Nt):
        for j in range(Np):
            z[i][j] = np.random.normal()

    r = mu*dt + z*sigma*np.sqrt(dt)
    s = np.zeros(shape=(NtP, Np))


    for i in range(Nt):
        for j in range(Np):
            s[i+1][j] = s[i][j]+r[i][j]


    price = s0*np.exp(s)
    terminal_value = price[Nt,:] -1

    return terminal_value

cpdef random_walk_prices(double mu, double sigma, double S0, int nt, int paths):
    """
    This function return the prices of each path.

    :param mu: mean | return
    :param sigma: std | risk
    :param S0: inicial price
    :param nt: Number of Time Steps
    :param paths: number of paths to simulate
    :return: retorna uma série de preços
    """
    mu = mu
    sigma = sigma
    s0 = S0

    Nt = nt
    Np = paths

    cdef double dt = 1/252
    cdef int NtP = Nt +1


    z = np.zeros(shape=(Nt, Np))

    cdef int i
    cdef int j
    for i in range(Nt):
        for j in range(Np):
            z[i][j] = np.random.normal()

    r = mu*dt + z*sigma*np.sqrt(dt)
    s = np.zeros(shape=(NtP, Np))


    for i in range(Nt):
        for j in range(Np):
            s[i+1][j] = s[i][j]+r[i][j]


    price = s0*np.exp(s)
    terminal_value = price[Nt,:] -1

    return price


cpdef autoregression2(double c0, double c1, double c2, int nt):
    """
    This function creates an montecarlo simulation based on an autoregression model.
    It returns a numpy array with an 2-lag autoregression serie.
    :param c0:
    :param c1:
    :param c2:
    :param nt: Number of time steps
    :return:
    """
    c0 = c0
    c1 = c1
    c2 = c2
    Nt = nt

    z = np.ndarray(shape=(Nt, 1))
    r = np.zeros(shape=(Nt, 1))

    cdef int t
    for t in range(Nt):
        z[t] = np.random.normal()


    for t in range(3, Nt):
        r[t] = c0 + c1 * r[t - 1] + c2 * r[t - 2] + z[t]

    return r


cpdef movingaverage2(double mu, double sigma, double phi1, double phi2, double phi3, int nt):

    """
    This function creates an montecarlo simulation based on a moving average model.
    It returns a numpy array with an 2-lag moving average serie.
    :param mu: average or return in finance
    :param sigma: standard deviation or risk in finance
    :param phi1: weight of the last observation in determing the next observation
    :param phi2: weight of the t-2 observation
    :param phi3: weight of the t-3 observation
    :param nt: number of time steps
    :return:
    """

    Mu = mu
    sigma = sigma
    phi1 = phi1
    phi2 = phi2
    phi3 = phi3
    Nt = nt


    z = np.ndarray(shape=(Nt, 1))
    r = np.zeros(shape=(Nt, 1))

    cdef int t

    for t in range(Nt):
        z[t] = np.random.normal()

    r[0] = Mu + sigma*z[0]
    r[1] = Mu + sigma*z[1] + phi1*z[0]
    r[2] = Mu + sigma*z[2] + phi2*z[1]  + phi1*z[0]

    for t in range(3, Nt):
        r[t] = Mu + sigma*z[t] + phi1*z[t-1] + phi2*z[t-2] + phi3*z[t-3]

    return r


def derivative(mu, sigma, s0, nt, path, p=1):
    """
    Essa função retorno o payoff de uma put ou call segundo o ativo subjacente.
    O caminho do ativo subjacente é trilhado segundo um modelo de monte carlo para o passeio aleatório.

    :param mu: média/retorno
    :param sigma: stdev/risco
    :param s0: preço inicial
    :param nt: passos temporais
    :param path: número de simulações
    :param p: Escolha entre put e call
    :return:
    """

    # Chamar a função random_walk_terminal para avaliar
    # a evolução do preço do ativo subjacente
    r = random_walk_terminal(mu, sigma, s0, nt, path)
    strike = 100

    # Escolha 1 para call e 0 para put
    if p == 1:
        t = r - strike
    elif p == 0:
        t = strike - r

    #  Opções não retornam menos que zero
    payoff = np.maximum(t, 0)
    return payoff