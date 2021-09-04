from linearReg import LinearReg

lr = LinearReg("linearX.csv", "linearY.csv", 0.075)

lr.run()

lr.plot()


