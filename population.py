from genome import Bird, IN_SIZE, HIDDEN_SIZE, OUT_SIZE
import numpy as np
import torch
from itertools import product

class BirdsPopulation:


    def __init__(self,pop_size=100,mutation_rate=0.05,seed=0):
        self.size = pop_size
        self.mr = mutation_rate
        self.population = self._init_population(self.size) 
        self.generation = 0
        if seed:
            np.random.seed(seed)
            torch.random.manual_seed(seed)
        else:
            np.random.seed()
            print(f'numpy seed - {np.random.get_state()[1][0]}')
            print(f'torch seed - {torch.random.seed()}')
            

    def _init_population(self,size):
        population = []
        for i in range(size):
            bird = Bird(i)
            # print(bird.state_dict())
            dna1 = torch.rand(HIDDEN_SIZE,IN_SIZE,requires_grad=False)
            # print(dna1)
            dna2 = torch.rand(OUT_SIZE,HIDDEN_SIZE,requires_grad=False)
            bird.load_state_dict({'dna.0.weight':dna1,'dna.2.weight':dna2})
            population.append(bird)
        return population

    
    def update_fitness(self,fitnesses):
        for bird_id,fitness,score in fitnesses:
            self.population[bird_id].fitness = fitness
            self.population[bird_id].score = score
    

    def select_fittest(self,top=10):
        sorted_pop = sorted(self.population,key=lambda b: b.fitness,reverse=True)
        return sorted_pop[:top] if top > 1 else sorted_pop[0] 
    

    def crossover(self):
        parents = self.select_fittest()
        offsprings = []
        counter = 0
        for p1,p2 in product(parents,parents):
            p1_dna = p1.state_dict()
            p2_dna = p2.state_dict()
            offspring = Bird(counter)
            counter += 1
            if np.random.choice([True, False]):
                dna = {'dna.0.weight':p1_dna['dna.0.weight'],'dna.2.weight':p2_dna['dna.2.weight']}
            else:
                dna = {'dna.0.weight':p2_dna['dna.0.weight'],'dna.2.weight':p1_dna['dna.2.weight']}
            offspring.load_state_dict(dna)
            offsprings.append(offspring)
        # del self.population[:]
        self.population = offsprings
    

    def mutate(self):
        for b in self.population:
            dna = b.state_dict()
            for k in dna:
                shape = dna[k].shape
                flat = torch.reshape(dna[k],(-1,))
                for i in range(len(flat)):
                    if np.random.rand() <= self.mr:
                        mutation = np.random.rand()
                        flat[i] += mutation if np.random.choice([True, False]) else -mutation
                dna[k] = torch.reshape(flat,shape)
            b.load_state_dict(dna)
    

    def load_population(self,generation=0,subdir=''):
        pop_data = torch.load(f'population_data/{subdir}/gen_{generation}') if subdir else torch.load(f'population_data/gen_{generation}')
        has_score = 'scores' in subdir or not subdir
        self.generation = pop_data['gen']
        self.size = pop_data['size']
        del self.population[:]
        self.population = [Bird(i) for i in range(self.size)]
        if has_score:
            for b,(dna,fitness,score) in zip(self.population, pop_data['population_state']):
                b.load_state_dict(dna)
                b.fitness = fitness
                b.score = score
        else:
            for b,(dna,fitness) in zip(self.population, pop_data['population_state']):
                b.load_state_dict(dna)
                b.fitness = fitness
    

    def save_population(self):
        pop_data = {'gen':self.generation,'size':self.size,'population_state':[(b.state_dict(),b.fitness,b.score) for b in self.population]}
        torch.save(pop_data,f'population_data/gen_{self.generation}')


    def next_gen(self):
        self.save_population()
        self.crossover()
        self.mutate()
        self.generation += 1
            
                
        
