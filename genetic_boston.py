import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

seed=2018
np.random.seed(seed)
random.seed(seed)

dataset=load_boston()
print('Boston features:\n', dataset.feature_names)
print('\nTotal no of Boston features: ', len(dataset.feature_names))
X,y=dataset.data,dataset.target
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8)
feature_names=dataset.feature_names
estimator=LinearRegression()
score=-1.0*cross_val_score(estimator,X,y,cv=5,scoring='neg_mean_squared_error')
estimator.fit(x_train,y_train)
a = np.mean(score)
print("\nError rate before optimization (Using Linear Regression):",a)

class GeneticFeatureSelection:
    def __init__(self,regressor,n_gen,n_best,size,n_rand,n_children,mutation_rate):
        self.regressor=regressor
        self.n_gen=n_gen
        self.n_best=n_best
        self.size=size
        self.mutation_rate=mutation_rate
        self.n_rand=n_rand
        self.n_children=n_children

    def initialize(self):
        population=[]
        for i in range(self.size):
            chromosome=np.ones(self.n_features,np.bool)
            mask=np.random.rand(len(chromosome))<0.3
            chromosome[mask]=False
            population.append(chromosome)
        return population

    def fitness(self,population):
        X,y=self.dataset
        fitness_scores=[]
        for chromosome in population:
            score=np.mean(-1.0*cross_val_score(self.regressor,X[:,chromosome],y,cv=5,scoring='neg_mean_squared_error'))
            fitness_scores.append(score)
        fitness_scores,population=np.array(fitness_scores),np.array(population)
        indices=np.argsort(fitness_scores)
        return list(fitness_scores[indices]),list(population[indices])

    def select(self,sorted_population):
        next_population=[]
        for i in range(self.n_best):
            next_population.append(sorted_population[i])
        for i in range(self.n_rand):
            next_population.append(random.choice(sorted_population))
        random.shuffle(next_population)
        return next_population

    def crossover(self,population):
        next_population=[]
        for i in range(int(len(population)/2)):
            for j in range(self.n_children):
                parent1,parent2=population[i],population[len(population)-1-i]
                offspring=parent1
                mask=np.random.rand(len(offspring))<0.5
                offspring[mask]=parent2[mask]
                next_population.append(offspring)
        return next_population

    def mutate(self,population):
        next_population=[]
        for i in range(len(population)):
            chromosome=population[i]
            if random.random()<self.mutation_rate:
                mask=np.random.rand(len(chromosome))<self.mutation_rate
                chromosome[mask]=False
            next_population.append(chromosome)
        return next_population

    def generate(self,population):
        sorted_fitness_scores,sorted_population=self.fitness(population)
        #print("\nFitness scores: ", sorted_fitness_scores)
        population=self.select(sorted_population)
        population=self.crossover(population)
        population=self.mutate(population)
        
        self.best_chromosomes.append(sorted_population[0])
        self.best_score.append(sorted_fitness_scores[0])
        self.best_avg.append(np.mean(sorted_fitness_scores))
        self.selected_features.append(sum(self.get_support()))
        print("Total number of features selected by GA: ", self.selected_features[-1])
        return population

    def fit(self,X,y):
        self.best_chromosomes,self.best_score,self.best_avg,self.selected_features=[],[],[],[]
        self.dataset=X,y
        self.n_features=X.shape[1]
        population=self.initialize()
        for i in range(self.n_gen):
        	print("\nGeneration: ", i+1)
        	population=self.generate(population)

    def get_support(self):
        return self.best_chromosomes[-1]

    def plot_progress(self):
        plt.plot(self.best_score,label="Best scores")
        plt.plot(self.best_avg,label="Best average")
        plt.xlabel('Generation')
        plt.ylabel('Fitness scores')
        plt.legend()
        plt.show()

    def plot_selections(self):
        plt.plot(self.selected_features)
        plt.xlabel("Generation")
        plt.ylabel("No. of selected features")
        plt.show()

selection=GeneticFeatureSelection(regressor=LinearRegression(),n_gen=15, size=200, n_best=40, n_rand=40,n_children=7, mutation_rate=0.05)
selection.fit(X,y)
selection.plot_progress()
selection.plot_selections()
score=-1.0*cross_val_score(estimator,X[:,selection.get_support()],y,cv=5,scoring='neg_mean_squared_error')
x_train,x_test,y_train,y_test=train_test_split(X[:,selection.get_support()],y,train_size=0.8)
estimator=LinearRegression()
estimator.fit(x_train,y_train)
#print(estimator.score(x_test,y_test))
b = np.mean(score)
print("\nError rate after optimization by Genetic Algorithm:",b)
print("\nError rate decreased by:", (a-b)*100/a, '%')