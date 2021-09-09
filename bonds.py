import numpy as np
import numpy_financial as npf
import numpy.linalg as linalg
from datetime import date
from workalendar.america import Brazil
cal = Brazil()


class Titulos:
    def __init__(self, taxa, start, end, c=0):
        """
        :param fv: Face Value
        :param c: coupon
        :param start: settlement date
        :param end: Maturity date
        """
        self.fv = 1000
        self.taxa = taxa
        self.coupon = c
        self.start = start
        self.end = end



    def ltn(self):
        """
        :param taxa: taxa contratada
        :param DU: Número de dias úteis
        :return:
        """
        DU = cal.get_working_days_delta(self.start, self.end)

        # Lidando com possível erro de entrada do usuário
        if self.taxa < 1:
            taxa = self.taxa
        elif self.taxa  > 1:
            taxa = self.taxa  / 100
        price = self.fv / (1 + taxa) ** (DU / 252)
        return price

    def ntn_f(self):
        """
        compute log returns for each ticker.

        parameters
         ----------
        taxa : scalar
        taxa contratada da ltn
        DU: list
        Dias úteis do pagamento de cada cupon

         returns
         -------
        preço : escalar
        preço de uma NTN-F
        """

        taxa = self.taxa
        # Lidando com possível erro de entrada do usuário
        if taxa < 1:
            taxa = taxa
        elif taxa > 1:
            taxa = taxa / 100
        # O Valor de Face sempre será 1000
        fv = 1000

        #Solucionar para as datas gerando cupons
        start = self.start
        end_year = date(start.year, 12, 31)
        start_next_year = date(start.year + 1, 1, 1)
        maturity = self.end

        #days é o restante do ano e days2 o restante do contrato após o ano seguinte
        days = cal.get_working_days_delta(start, end_year)
        days2 = cal.get_working_days_delta(start_next_year, maturity)

        if days < 252 / 2:
            coupon = 1
        else:
            coupon = 2

        number_coupons = int((days2 / 252) * 2) + coupon
        # Criamos uma np.array para facilitar manipulação
        DU = np.zeros(number_coupons)

        for i in range(1, number_coupons):
            DU[0] = days
            DU[i] = DU[i - 1] + 252 / 2

        # O valor do cupom é fixo e é melhor aproximado por essa conta
        c = fv * ((1 + 0.1) ** (1 / 2) - 1)

        # Manipulação para a criação do fluxo de caixa
        terminal = fv + c
        fluxo_caixa = np.full(len(DU), fill_value=c)
        fluxo_caixa[-1] = terminal
        dcf = np.full(len(DU), fill_value=taxa)
        price = sum(fluxo_caixa / (1 + dcf) ** (DU / 252))
        return price

    def ntn_b_principal(self, VNA, IPCA):
        """
        Essa função retorna o preço de uma NTN-B
        :param taxa: Na hora da negociação
        :param VNA: VNA na hora da negociação
        :param DU = Dias úteis até o vencimento
        :param IPCA: IPCA projetado. Deve estar em percentual
        :param date: Data em que a negociação será feita
        :return: preço da NTN-B
        """
        DU = self.end - self.start
        DU = DU.days

        my_date = self.start
        taxa = self.taxa
        # Lidando com possível erro de entrada do usuário
        if taxa < 1:
            taxa = taxa
        elif taxa > 1:
            taxa = taxa / 100

        IPCA = IPCA / 100

        # Lidando com o tempo e seus casos extremos
        # Esse é um ponto crítico no cálculo da NTN-B
        if my_date.month == 1:
            last = datetime.date(my_date.year - 1, 12, 15)
        else:
            last = datetime.date(my_date.year, my_date.month - 1, 15)

        if my_date.day < 15:
            this_month = datetime.date(my_date.year, my_date.month, 15)
        else:
            this_month = my_date

        if my_date.month == 12:
            next_month = datetime.date(my_date.year + 1, 1, 15)
        else:
            next_month = datetime.date(my_date.year, my_date.month + 1, 15)

        # A partir daqui come;a o real cálculo da ntn-b

        # Se o cálculo é para o DIA 15, o VNA náo precisa ser calculado
        if my_date.day == 15:
            VNA_p = VNA
        else:
            pr = (my_date - last) / (next_month - this_month)
            VNA_p = VNA * (1 + IPCA) ** (pr)

        #Calculando a cotação
        cotacao = 1 / (1 + taxa) ** (DU / 252)

        #preço de compra da NTN-B
        valor = VNA_p * cotacao
        return valor

    def ntn_b(self, VNA, DU, IPCA, date):
        """
        Essa função retorna o preço de uma NTN-B
        :param taxa: Na hora da negociação
        :param VNA: VNA na hora da negociação
        :param DU = Lista de dias úteis de cada cupom
        :param IPCA: IPCA projetado. Deve estar em percentual
        :param date: Data em que a negociação será feita
        :return: preço da NTN-B
        """
        taxa = self.taxa
        # Lidando com possível erro de entrada do usuário
        if taxa < 1:
            taxa = taxa
        elif taxa > 1:
            taxa = taxa / 100

        # Criamos uma np.array para facilitar manipulação
        DU = np.array(DU)

        # Temos que normalizar o IPCa
        IPCA = IPCA / 100

        # Lidando com o tempo e seus casos extremos
        # Esse é um ponto crítico no cálculo da NTN-B
        my_date = date
        if my_date.month == 1:
            last = datetime.date(my_date.year - 1, 12, 15)
        else:
            last = datetime.date(my_date.year, my_date.month - 1, 15)

        if my_date.day < 15:
            this_month = datetime.date(my_date.year, my_date.month, 15)
        else:
            this_month = my_date

        if my_date.month == 12:
            next_month = datetime.date(my_date.year + 1, 1, 15)
        else:
            next_month = datetime.date(my_date.year, my_date.month + 1, 15)

        # A partir daqui começa o real cálculo da ntn-b

        pr = (my_date - last) / (next_month - this_month)

        # Calculando e controlando o VNA projetado
        if my_date.day == 15:
            VNA_p = VNA
        else:
            VNA_p = VNA * (1 + IPCA) ** (pr)

        #Calculando a cotação que inclui os cupons
        c = ((1 + 0.06) ** (1 / 2) - 1)
        terminal = 1
        fluxo_caixa = np.full(len(DU), fill_value=c)
        fluxo_caixa[-1] = fluxo_caixa[-1] + terminal
        dcf = np.full(len(DU), fill_value=taxa)
        cotacao = sum(fluxo_caixa / (1 + dcf) ** (DU / 252))

        #preço de compra da NTN-B
        valor_final = VNA_p * cotacao
        return valor_final

    def lft(self, taxa, DU, VNA, selic):
        """
        Retorna o preço de compra de uma LFT
        :param taxa: taxa contratada com ágio ou deságio
        :param DU: dias úteis
        :param VNA: VNA corrigido pela SELIC
        :param selic: Taxa Selic projetada e anualizada
        :return:
        """
        if selic < 1:
            selic = selic
        elif taxa > 1:
            selic = selic / 100

        VNA_p = VNA * (1 + selic) ** (1 / 252)
        cotacao = 1 / (1 + taxa) ** (DU / 252)
        return VNA_p * cotacao

    def bondPrice(self, ttm: int , ytm):
        """
        param:
        ttm = Time to maturity
        ytm = Yield to Maturity
        """
        c = self.coupon
        fv = self.fv
        cashFlow = []
        if c < 0:
            c = c
        else:
            c = c/100
        if ytm < 0:
            ytm = ytm
        else:
            ytm = ytm/100

        [cashFlow.append((fv*c)/(1+ytm)**(i)) for i in range(1,ttm)]
        cashFlow.append((fv+(fv*c))/(1+ytm)**(ttm))
        price = sum(cashFlow)
        return price



    def ytm(self, r):
        """
        This function return the prices of each path.
    
        :param fv: Bonds Face Value
        :param rates: a list of interest rates
        :param c: coupon rate
        :return: Bond Yield to Maturity and Price
        """

        #Setting ttm = Time to maturity
        ttm = len(r)

        fv = self.fv
        c = self.coupon

        if c > 1:
            c = c/100
        else:
            c = c

        ttm1 = ttm-1

        #Creating a coupon array
        cashF = np.array([fv*c]*ttm1)
        cashF = np.append(cashF,(fv*c)+fv)

        #Create a array with zeros to fill with discounted factors

        dcf = np.zeros(ttm, dtype=np.double)
        for i in range(0,ttm):
            dcf[i] = cashF[i] * (1/(1+r[i])**(i+1))

        # Fiding prices
        price = np.sum(dcf)

        #Creating cash flow structure to calculate YTM
        cashF = np.insert(cashF, 0, -price)
        ytm = npf.irr(cashF)
        Bond_Characteristics = { "Bond price": price,
                                "Bond Yield":round(ytm*100, 3),

        }
        return Bond_Characteristics

    def bondPriceElasticity(self, rates, delta):
        """
        :param fv: Bonds Face Value
        :param rates: a list of interest rates
        :param c: coupon rate
        :param delta: change in Yield to Maturity
        :return: Bond Yield to Maturity and Price
        """

        fv = self.fv
        c = self.coupon
        ttm = len(rates)

        values = ytm(rates)
        price = values['Bond price']
        rates = np.array(rates)
        rates = rates*(1/100)
        c = c/100
        delta = delta/100
        ttm1 = ttm-1
        cashF = np.vstack([fv*c]*ttm1)
        cashF = np.append(cashF,(fv*c)+fv)
        dcf = np.array([])
        for i in range(0,ttm):
            dcf = np.append(dcf,cashF[i]*(i+1)/(1+rates[i])**(i+2) )
        pe_factor = np.sum(dcf)
        b = -1/price
        price_elasticity = b*pe_factor

        delta_bond = -price*abs(price_elasticity)*delta
        bond_elasticity = {'Bond Price': price,
                           'Bond new price': price+delta_bond,
                            'Bond Elasticity':price_elasticity,
                            'Price Change': delta_bond,

        }
        return bond_elasticity




    def convexity(self, r):
        """
        param:
        fv = Face Value
        ttm = Time to maturity
        c = coupon
        ytm = Yield to Maturity
        """

        fv = self.fv
        c = self.coupon
        ytm = ytm(r)
        ttm = len(r)

        price = bondPrice(fv, ttm, c, ytm)
        c = c/100
        ytm = ytm/100
        x = []
        [x.append((fv*c)/(1+ytm)**(i)) for i in range(1,ttm)]
        x.append((fv+(fv*c))/(1+ytm)**(ttm))

        y = []
        [y.append(x[i] * (i+1) * (i+2)  ) for i in range(0,ttm)]
        dfc = sum(y)
        cx = (dfc*0.5)/(((1+ytm)**2)*price)
        return cx

    def mac_mod_cx_duration(self, r):
        """
        param:
        fv = Face Value
        ttm = Time to maturity; If the bond is a perpetuity, ttm =0
        c = coupon
        ytm = Yield to Maturity
        """

        fv = self.fv
        c = self.coupon
        ytm = ytm(r)
        ttm = len(r)

        price = bondPrice(fv, ttm, c, ytm)
        cx = convexity(fv,ttm,c, ytm)
        c = c/100
        ytm = ytm/100
        x =[]
        if ttm == 0:
            modD = 1/ytm
            D = modD*(1+ytm)
        else:
            if c == 0:
                D = ttm
            else:
                [x.append((fv*c)/(1+ytm)**(i) *i) for i in range(1,ttm)]
                x.append((fv+(fv*c))/(1+ytm)**(ttm) * ttm)
                d = sum(x)
                D = d/price

        modD = D/(1+ytm)
        bond_cara = {'Price': price,
                      'Bond Macauly Duration':D,
                      'Modified Duration': modD,
                      'Convexity': cx,

        }

        return bond_cara





    def bondRisk(self, r, delta):
        """
        param:
        fv = Face Value
        ttm = Time to maturity; If the bond is a perpetuity, ttm =0
        c = coupon
        ytm = Yield to Maturity
        delta = change in yield to maturity
        """
        fv = self.fv
        ttm = len(r)
        c = self.coupon
        ytm = ytm(r)


        delta = delta/100
        values = mac_mod_cx_duration(fv,ttm,c,ytm)
        price = values['Price']
        D = values['Bond Macauly Duration']
        modD = values['Modified Duration']
        cx = values['Convexity']

        """Now we calculate the change in price of this bond.
        delta_price = is the approximation only with modifie duration
        delta_price2 = is the approximation with modifie duration and convexity
        """
        delta_price = -price*modD*delta
        delta_price2 = price*(-modD*delta + cx*delta**2)
        """ Now we use the change to see what is the new price in these two cases. l
        """
        newprice = price + delta_price
        newprice2 = price + delta_price2
        bond_risks = {'Bond Macauly Duration':D,
                       'Bond Modified Duration':modD,
                       'first order change in price': delta_price,
                      'Second order change in price':delta_price2,
                      'Original Price':price,
                      'New price Mod Duration': newprice,
                      'New price Mod and Convexity': newprice2,
        }
        return bond_risks



    def arbitrageEstrategy(fv, price, c, ttm):
        """
        param:
        fv = Face Value
        ttm = List with Time to Maturity.
        c = List with Coupon interest;
        ytm = List with Yield to Maturity.
        """
        price = np.array(price)
        price = price * (-1)

        fv = np.array(fv)
        c = np.array(c)
        c = c/100

        ttm = np.array(ttm)

        # The shape of the Matrix. This will depend of the bond with the higher time to maturity.
        mformat = np.max(ttm)
        ttmz = ttm-1

        # Manipulação das constantes para a estratégia
        cons = np.array([])
        len = mformat
        cons = np.vstack([0]*(len))
        cons = np.insert(cons, 0, 100)

        #creating a Matrix
        Matrix = np.zeros((price.size, mformat))


         # Setting the Matrix
        #Setting the o coupon bonds
        for i in range(0, price.size):
            if (c[i] == 0):
                Matrix[i][ttmz[i]] = fv
            else:
                pass

        #Setting the coupon payment
        for i in range(0, price.size):
            if (np.max(Matrix[i][:]) == 0):
                for j in range(0, mformat):
                    Matrix[i][j] = (fv*c[i])

        #Setting the principal payment
        for i in range(0, price.size):
            for j in range(0, mformat):
                if(c[i] != 0 and j == ttmz[i]):
                    Matrix[i][j] = fv+(fv*c[i])

       #Cleaning the Matrix
        for i in range(0, price.size):
            for j in range(0, mformat):
                if(c[i] != 0 and j > ttmz[i]):
                    Matrix[i][j] = 0

        #Get together prices array and matrix
        matrix = np.column_stack((price, Matrix))

        #Transposing the Matrix
        matrix = np.transpose(matrix)

        #Solving the Matrix. Maybe another function for non-quadratic matrix
        answer = linalg.solve(matrix, cons)

        return answer
