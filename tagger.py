import sys
import math
import numpy as np

train_in = sys.argv[1]
val_in = sys.argv[2]
test_in = sys.argv[3]
train_out = sys.argv[4]
test_out = sys.argv[5]
metrics_out = sys.argv[6]
epoch = int(sys.argv[7])
flag = int(sys.argv[8])


class Model:
    def __init__(self):
        self.f_name = ""
        self.all_word = []  # Save all the words
        self.all_label = []  # Save all the labels
        self.Word = []  # Save all the possible words
        self.Label = []  # Save all the possible labels
        self.F = []  # Save nr of "1" in self.Word
        self.theta = np.zeros(1, dtype=int)  # Save the weight numbers

        self.s = (0, 0)

        # Model Parameter
        self.cur = 0
        self.prev = 0
        self.nex = 0

        self.flag = 0

        self.Exp = []
        self.exp = 0
        self.sigma_exp = 0
        self.par = 0

        self.log_likelihood = 0
        self.blank = 0
        self.epoch = -1

        self.prob = 0
        self.K = 0
        self.wrong = 0

    def read_file(self, f_name):  # Read a File to get all words and their labels
        self.f_name = f_name
        self.all_word.append('')  # For Model 2
        self.all_label.append('')  # Match with self.all_word
        db = open(self.f_name, 'r+').read().splitlines()
        for i in range(0, len(db)):
            db[i] = db[i].split('\t')
            self.all_word.append(db[i][0])
            if db[i][0] != '':
                self.all_label.append(db[i][1])
            else:
                self.all_label.append('')
        self.all_word.append('')  # For Model 2
        self.all_label.append('')  # Match with self.all_word

    def map(self):  # Get all possible words and labels from training data
        self.flag = flag  # Get Model Information

        self.Word.append(self.all_word[0])
        self.Label.append(self.all_label[0])
        for i in range(1, len(self.all_word)):
            if self.all_word[i] not in self.Word:
                self.Word.append(self.all_word[i])
            if self.all_label[i] not in self.Label:
                self.Label.append(self.all_label[i])
        self.Word.sort()
        self.Label.sort()
        self.Word.remove('')  # Length of Word will be the Length of each row of Theta (M)
        self.Label.remove('')  # Length of Label will be the number of rows of Theta (K)

        if self.flag == 2:  # Model 2
            self.Word.append('BOS')
            self.Word.append('EOS')

        self.s = (len(self.Label), (len(self.Word) + 1))  # Generation of the Initial Theta
        self.theta = np.zeros(self.s, dtype=float)

    def fea(self):
        for i in range(1, len(self.all_word) - 1):
            if self.all_word[i] == '':
                self.F.append([])
                self.blank += 1
            else:
                if self.flag == 1:  # Model 1
                    for m in range(0, len(self.Word)):
                        if self.all_word[i] == self.Word[m]:
                            self.cur = m+1
                    self.F.append([self.cur])
                else:  # Model 2
                    for m in range(0, len(self.Word)):
                        if self.all_word[i-1] == self.Word[m]:
                            self.prev = m+1
                        if self.all_word[i] == self.Word[m]:
                            self.cur = m+1
                        if self.all_word[i+1] == self.Word[m]:
                            self.nex = m+1
                    self.F.append([self.cur, self.prev, self.nex])

    def learning(self, ep):  # One Iteration for Theta Update
        for i in range(1, len(self.all_word)-1):  # Don't see the BOS and EOS of the whole data set
            if self.all_word[i] == '':
                if self.epoch == ep:
                    if self.f_name == train_in and self.epoch == ep:
                        f = open(train_out, 'a')
                        f.write('\n')
                        f.close()

                    if self.f_name == test_in and self.epoch == ep:
                        f = open(test_out, 'a')
                        f.write('\n')
                        f.close()
            else:
                for k in range(0, len(self.Label)):  # Computation of the of the prob
                    if self.flag == 1:
                        self.exp = math.exp(self.theta[k, 0] + self.theta[k, (self.F[i-1][0])])
                    else:
                        self.exp = math.exp(self.theta[k, 0] + self.theta[k, (self.F[i-1][0])]
                                            + self.theta[k, (self.F[i-1][1])] + self.theta[k, (self.F[i-1][2])])
                    self.Exp.append(self.exp)
                    self.sigma_exp += self.exp
                if self.f_name == train_in:
                    for k in range(0, len(self.Label)):  # Generation of all the gradients
                        if self.epoch < ep:
                            if self.all_label[i] == self.Label[k]:
                                self.par = -(1 - float(self.Exp[k])/self.sigma_exp)
                                self.log_likelihood += - np.log(float(self.Exp[k])/self.sigma_exp)/(len(self.F) - self.blank)
                            else:
                                self.par = -(- float(self.Exp[k])/self.sigma_exp)
                            self.theta[k, 0] -= self.par  # Update of the theta
                            self.theta[k, (self.F[i-1][0])] -= self.par
                            if self.flag == 2:
                                self.theta[k, (self.F[i - 1][1])] -= self.par
                                self.theta[k, (self.F[i - 1][2])] -= self.par
                        else:
                            p = float(self.Exp[k]) / self.sigma_exp
                            if p > self.prob:
                                self.prob = p
                                self.K = k

                    if self.epoch == ep:
                        if i == 1:
                            f = open(train_out, 'w')
                        else:
                            f = open(train_out, 'a')
                        f.write(self.Label[self.K] + '\n')
                        f.close()

                        if self.Label[self.K] != self.all_label[i]:
                            self.wrong += 1
                elif self.f_name == val_in:  # Only compute likelihood for validation set
                    for k in range(0, len(self.Label)):  # Generation of all the gradients
                        if self.all_label[i] == self.Label[k]:
                            self.log_likelihood += - np.log(float(self.Exp[k])/self.sigma_exp)/(len(self.F) - self.blank)
                else:
                    for k in range(0, len(self.Label)):
                        p = float(self.Exp[k]) / self.sigma_exp
                        if p > self.prob:
                            self.prob = p
                            self.K = k

                    if self.epoch == ep:
                        if i == 1:
                            f = open(test_out, 'w')
                        else:
                            f = open(test_out, 'a')
                        f.write(self.Label[self.K] + '\n')
                        f.close()

                        if self.Label[self.K] != self.all_label[i]:
                            self.wrong += 1

                self.Exp = []
                self.sigma_exp = 0
                self.prob = 0

        self.epoch += 1

    def likelihood(self):
        if self.epoch == 1 and self.f_name == train_in:
            f = open(metrics_out, 'w')
        else:
            f = open(metrics_out, 'a')

        if self.epoch >= 1:
            if self.f_name == train_in:
                f.write("epoch=" + str(self.epoch) + " likelihood(train): " + str(round(self.log_likelihood, 6)) + '\n')
                f.close()
            else:
                f.write("epoch=" + str(self.epoch) + " likelihood(validation): " + str(round(self.log_likelihood, 6)) + '\n')
                f.close()
        f.close()
        self.log_likelihood = 0

    def error(self):
        f = open(metrics_out, 'a')
        er = float(self.wrong)/(len(self.F) - self.blank)
        if self.f_name == train_in:
            f.write("error(train): " + str(round(er, 6)) + '\n')
        else:
            f.write("error(test): " + str(round(er, 6)) + '\n')
        f.close()

# Ready for All the files
tr = Model()
va = Model()
te = Model()

tr.read_file(train_in)
va.read_file(val_in)
te.read_file(test_in)

tr.map()

# Set validation Set and Test Set in the same way as Training Set
va.Word = tr.Word
va.Label = tr.Label
va.flag = tr.flag

te.Word = tr.Word
te.Label = tr.Label
te.flag = tr.flag

# Extract Features
tr.fea()
va.fea()
te.fea()

for e in range(1, epoch+2):  # Calculation of Likelihood
    tr.learning(epoch)
    tr.likelihood()

    if e < epoch+1:
        va.theta = tr.theta
    va.learning(epoch)
    va.likelihood()

tr.learning(epoch)
tr.error()  # Calculation of the training error

te.theta = tr.theta
te.epoch = epoch
te.learning(epoch)
te.error()  # Calculation of the test error
