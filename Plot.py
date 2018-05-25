import matplotlib.pyplot as plt

# 1.2.1

x = []
f_1 = open("train_y_1.txt", 'r+').read().splitlines()
f_2 = open("va_y_1.txt", 'r+').read().splitlines()

for i in range(0, len(f_1)):
    x.append(i)
    f_1[i] = float(f_1[i])
    f_2[i] = float(f_2[i])

print f_1
print f_2

plt.title('Negative Log Likelihood of Data for Model 1')
plt.xlabel('Number of Epoch')
plt.ylabel('Negative Log Likelihood')

plt.plot(x, f_1, color="red", linewidth=2.5, linestyle="-", label="training data")
# plt.plot(x, f_1, 'ro', color='black')
plt.plot(x, f_2, color="blue", linewidth=2.5, linestyle="-", label="validation data")
# plt.plot(x, f_2, 'ro', color='black')
plt.legend(loc='upper right')
plt.axis([0, 100, 0.32, 0.55])
plt.show()

# 1.2.2
x = []
f_3 = open("train_y_2.txt", 'r+').read().splitlines()
f_4 = open("va_y_2.txt", 'r+').read().splitlines()

for i in range(0, len(f_3)):
    x.append(i)
    f_3[i] = float(f_3[i])
    f_4[i] = float(f_4[i])

print f_3
print f_4

plt.title('Negative Log Likelihood of Data for Model 2')
plt.xlabel('Number of Epoch')
plt.ylabel('Negative Log Likelihood')

plt.plot(x, f_3, color="red", linewidth=2.5, linestyle="-", label="training data")
# plt.plot(x, f_1, 'ro', color='black')
plt.plot(x, f_4, color="blue", linewidth=2.5, linestyle="-", label="validation data")
# plt.plot(x, f_2, 'ro', color='black')
plt.legend(loc='upper right')
plt.axis([0, 100, 0.05, 0.22])
plt.show()