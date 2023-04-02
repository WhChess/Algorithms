import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

data = pandas.read_csv('cost_revenue_clean.csv')
X = DataFrame(data, columns=['production_budget_usd'])
y = DataFrame(data, columns=['worldwide_gross_usd'])

regression = LinearRegression()
regression.fit(X, y) # Regression line çiziyoruz. Teoride karelerin azaltılması olarak geçer.
print("theta_1 (Eğim): ",regression.coef_) # Eğim
print("theta_0 (Sabit): ",regression.intercept_) # Sabit

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.3)
plt.plot(X, regression.predict(X), color='orange', linewidth=4) # Çizgiyi grafik üzerinde gösteriyoruz.
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0,3000000000)
plt.xlim(0,450000000)

print("regression.score = ",regression.score(X, y)) # r-square değerini hesaplıyoruz. Detayına teoride girilecek.
print("Bu model varyasyonun yaklaşık %55'ini açıklayabilir.")
print("Bu problemin sonuçlarının yaklaşık %45'i başka nedenlere bağlı olarak gelişir.")

plt.show()
