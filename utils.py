import matplotlib.pyplot as plt
import numpy as np 

def plot_fittest(gens,birds,subdirs):
    results = []
    for subdir in subdirs:
        fitnesses = []
        split = subdir.split('-')
        for gen in gens:
            birds.load_population(generation=gen,subdir=subdir)
            fitnesses.append(birds.select_fittest(1).fitness)
        plt.plot(gens,fitnesses)
        plt.title(f'Fittest birds - population_size={split[1]}, mutation_rate={split[2]}%')
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness of Fittest Bird')
        plt.tight_layout()
        # plt.savefig(f'plots/top_birds_{gens[0]}-{gens[-1]}_{split[2]}.png')
        plt.close()
        results.append(fitnesses)
    return results

def plot_parents(gens,birds,subdirs):
    results = []
    for subdir in subdirs:
        fitnesses = []
        split = subdir.split('-')
        for gen in gens:
            birds.load_population(generation=gen,subdir=subdir)
            fitnesses.append(np.mean([b.fitness for b in birds.select_fittest()]))
        plt.plot(gens,fitnesses)
        plt.title(f'Fitness of Parent Birds - population_size={split[1]}, mutation_rate={split[2]}%')
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness of Parents')
        plt.tight_layout()
        # plt.savefig(f'plots/parents_fitness_{gens[0]}-{gens[-1]}_{split[2]}.png')
        plt.close()
        results.append(fitnesses)
    return results

def plot_pop_fitness(gens,birds,subdirs):
    results = []
    for subdir in subdirs:
        fitnesses = []
        split = subdir.split('-')
        for gen in gens:
            birds.load_population(generation=gen,subdir=subdir)
            fitnesses.append(np.mean([b.fitness for b in birds.population]))
        plt.plot(gens,fitnesses)
        plt.title(f'Population Fitness - population_size={split[1]}, mutation_rate={split[2]}%')
        plt.xlabel('Generation')
        plt.ylabel('Average Population Fitness')
        plt.tight_layout()
        # plt.savefig(f'plots/pop_fitness_{gens[0]}-{gens[-1]}_{split[2]}.png')
        plt.close()
        results.append(fitnesses)
    return results

def plot_fitness_comparison(gens,top,mean,parents,pop_size,mr):
    plt.plot(gens,top,label='Top Bird')
    plt.plot(gens,parents,label='Top 10 Birds')
    plt.plot(gens,mean,label='All Birds')
    plt.title(f'Fitness comparison - population_size={pop_size}, mutation_rate={mr}%')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f'plots/comp_fitness_{gens[0]}-{gens[-1]}_{mr}.png')
    plt.close()

def plot_highscores(gens,birds,subdirs):
    results = []
    for subdir in subdirs:
        if 'scores' not in subdir:
            continue
        fitnesses = []
        split = subdir.split('-')
        for gen in gens:
            birds.load_population(generation=gen,subdir=subdir)
            fitnesses.append(birds.select_fittest(1).score)
        plt.plot(gens,fitnesses)
        plt.title(f'Fittest birds - population_size={split[1]}, mutation_rate={split[2]}%')
        plt.xlabel('Generation')
        plt.ylabel('Average Score of Fittest Bird')
        plt.tight_layout()
        # plt.savefig(f'plots/top_birds_score_{gens[0]}-{gens[-1]}_{split[2]}.png')
        plt.close()
        results.append(fitnesses)
    return results

def plot_parents_score(gens,birds,subdirs):
    results = []
    for subdir in subdirs:
        if 'scores' not in subdir:
            continue
        fitnesses = []
        split = subdir.split('-')
        for gen in gens:
            birds.load_population(generation=gen,subdir=subdir)
            fitnesses.append(np.mean([b.score for b in birds.select_fittest()]))
        plt.plot(gens,fitnesses)
        plt.title(f'Score of Parent Birds - population_size={split[1]}, mutation_rate={split[2]}%')
        plt.xlabel('Generation')
        plt.ylabel('Average Score of Parents')
        plt.tight_layout()
        # plt.savefig(f'plots/parents_score_{gens[0]}-{gens[-1]}_{split[2]}.png')
        plt.close()
        results.append(fitnesses)
    return results

def plot_pop_score(gens,birds,subdirs):
    results = []
    for subdir in subdirs:
        if 'scores' not in subdir:
            continue
        fitnesses = []
        split = subdir.split('-')
        for gen in gens:
            birds.load_population(generation=gen,subdir=subdir)
            fitnesses.append(np.mean([b.score for b in birds.population]))
        plt.plot(gens,fitnesses)
        plt.title(f'Population Score - population_size={split[1]}, mutation_rate={split[2]}%')
        plt.xlabel('Generation')
        plt.ylabel('Average Population Score')
        plt.tight_layout()
        # plt.savefig(f'plots/pop_score_{gens[0]}-{gens[-1]}_{split[2]}.png')
        plt.close()
        results.append(fitnesses)
    return results

def plot_score_comparison(gens,top,mean,parents,pop_size,mr):
    plt.plot(gens,top,label='Top Bird')
    plt.plot(gens,parents,label='Top 10 Birds')
    plt.plot(gens,mean,label='All Birds')
    plt.title(f'Score comparison - population_size={pop_size}, mutation_rate={mr}%')
    plt.xlabel('Generation')
    plt.ylabel('Average Score')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f'plots/comp_score_{gens[0]}-{gens[-1]}_{mr}.png')
    plt.close()

def plot_comparison(gens,p1,p2,p3,title,ylabel,suffix):
    plt.plot(gens,p1[0],label=p1[1])
    plt.plot(gens,p2[0],label=p2[1])
    plt.plot(gens,p3[0],label=p3[1])
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel(ylabel)
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f'plots/comp_{gens[0]}-{gens[-1]}_{suffix}.png')
    plt.close()

