import time
import flappy_bird_gym
from population import BirdsPopulation
import numpy as np
import multiprocessing
import utils
import torch


birds = BirdsPopulation(rgb=False)
envs = {bird.id: flappy_bird_gym.make("FlappyBird-v0") for bird in birds.population}
main_env = flappy_bird_gym.make("FlappyBird-v0")


def run_with_render(bird, recording=False):
    print(f'running with render - bird{bird.id} with fitness of {bird.fitness}')
    obs = main_env.reset()
    
    if recording:
        main_env.render()
        input('ready to record ?')

    score,fitness = 0,0
    while True:
        action = bird(obs)
        # Processing:
        obs, reward, done, info = main_env.step(action)
        fitness += reward
        # Rendering the game:
        main_env.render()
        time.sleep(1 / 60)  # FPS
        score = info['score']
        # Checking if the player is still alive
        if done:
            break
    print(f'Fitness = {fitness}, Score = {score}')
    main_env.close()


def evaluate_fitness(bird):
    env = envs[bird.id]
    runs = []
    scores = []
    print(f'{bird.id}: started!')
    for _ in range(3):
        obs = env.reset()
        fitness = 0
        while True:
            action = bird(obs)
            obs, reward, done, info = env.step(action)
            fitness += reward

            if done:
                scores.append(info['score'])
                break
        env.close()
        runs.append(fitness)
    print(f'{bird.id}: done!')
    return bird.id, np.mean(runs), np.mean(scores)


def evolution(initial_gen=0):
    max_fitness = 0
    max_gen = 0
    max_bird = 0
    max_score = 0
    stime = time.time()
    for gen in range(initial_gen, initial_gen+101):
        with multiprocessing.get_context('fork').Pool(8) as pool:
            fitnesses = pool.map(evaluate_fitness,birds.population)
            birds.update_fitness(fitnesses)
            bird_id, fitness, score = sorted(fitnesses,key=lambda x: x[1],reverse=True)[0]
            print(f'Gen:{gen} fittest is BIRD-{bird_id} with fittnes of {fitness} and score of {score}')
            if fitness > max_fitness:
                max_fitness = fitness
                max_bird = bird_id
                max_gen = gen
                max_score = score

        if gen % 10 == 0:
            print(f'Best fitness so far: {max_fitness} with score of {max_score}, by BIRD-{max_bird} at gen:{max_gen}')
        birds.next_gen()
    print(f'running all generations took: {time.time()-stime}s')

if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(
            prog='main.py', description='Neuroevolution for flappy bird')
        parser.add_argument('-train', dest='train', action='store_true', help='use this flag to run training')
        parser.add_argument('-ig', dest='initial_gen', metavar='G', default=0, type=int, help='Initial generation')
        parser.add_argument('-load', dest='load', action='store_true', help='use this flag to load population from initial_gen if initial_gen is 0.')
        parser.add_argument('-plot', dest='plot', action='store_true', help='use this flag to plot best of gen 0 to 200')
        parser.add_argument('-sd', dest='subdir', metavar='SUBDIR', default='', type=str, help='subdir for weights')
        args = parser.parse_args()
        return args
    args = parse_args()
    if args.initial_gen > 0 or args.load:
        birds.load_population(generation=args.initial_gen,subdir=args.subdir)
    if args.train:
        evolution(args.initial_gen)
    elif not args.plot:
        bird = birds.select_fittest(1)
        print(
            f'Running single run using BIRD-{bird.id} from generation #{birds.generation}')

        run_with_render(bird, recording=True)
    else:
        subdirs = ['100-100-30-scores','100-100-10-scores','100-100-5-scores'] 
        gens = range(101)
        top = utils.plot_fittest(gens,birds,subdirs)
        mean = utils.plot_pop_fitness(gens,birds,subdirs)
        parents = utils.plot_parents(gens,birds,subdirs)
        for t,m,p,sd in zip(top,mean,parents,subdirs):
            split = sd.split('-')
            utils.plot_fitness_comparison(gens,t,m,p,split[1],split[2])
        top_score = utils.plot_highscores(gens,birds,subdirs)
        mean_score = utils.plot_pop_score(gens,birds,subdirs)
        parents_score = utils.plot_parents_score(gens,birds,subdirs)
        for t,m,p,sd in zip(top_score,mean_score,parents_score,filter(lambda x: 'scores' in x,subdirs)):
            split = sd.split('-')
            utils.plot_score_comparison(gens,t,m,p,split[1],split[2])
        utils.plot_comparison(gens,(top_score[0],'Mutation rate 30%'),(top_score[1],'Mutation rate 10%'),(top_score[2],'Mutation rate 5%'),'Comparison with different mutation rates','Average Score (Fittest bird)','score_mr')
        utils.plot_comparison(gens,(top[0],'Mutation rate 30%'),(top[1],'Mutation rate 10%'),(top[2],'Mutation rate 5%'),'Comparison with different mutation rates','Average Fitness (Fittest bird)','fitness_mr')
        utils.plot_comparison(gens,(mean[0],'Mutation rate 30%'),(mean[1],'Mutation rate 10%'),(mean[2],'Mutation rate 5%'),'Comparison with different mutation rates','Average Fitness (Entire population)','fitness_pop_mr')
        utils.plot_comparison(gens,(mean_score[0],'Mutation rate 30%'),(mean_score[1],'Mutation rate 10%'),(mean_score[2],'Mutation rate 5%'),'Comparison with different mutation rates','Average Score (Entire population)','score_pop_mr')
        utils.plot_comparison(gens,(parents[0],'Mutation rate 30%'),(parents[1],'Mutation rate 10%'),(parents[2],'Mutation rate 5%'),'Comparison with different mutation rates','Average Fitness (Top 10 birds)','fitness_parents_mr')
        utils.plot_comparison(gens,(parents_score[0],'Mutation rate 30%'),(parents_score[1],'Mutation rate 10%'),(parents_score[2],'Mutation rate 5%'),'Comparison with different mutation rates','Average Score (Top 10 birds)','score_parents_mr')
        
